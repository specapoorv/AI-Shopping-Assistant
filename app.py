# External Packages:
from __future__ import annotations
import gradio as gr
import os
from dataclasses import dataclass
from typing import List, Tuple, Union
from PIL import Image

# Importing from our various other files:
import my_clipmlp
import yolo_shoe_detection
import llm
from llm import beginning_llm_prompt, second_llm_prompt, clip_category_prompt, clip_preferences_prompt
myLLM = llm.MyLLMClass()

LocalImg = Union[str, Image.Image]

@dataclass
class Product:
    image: LocalImg
    price: str
    name: str = ""


# -----------------------------------------------------------------------------
# Core Backend Calling
# -----------------------------------------------------------------------------

def generate_initial_recommendations(img_path: str | None) -> Tuple[List[Product], str]:
    image = Image.open(img_path).copy()
    cropped_images = yolo_shoe_detection.get_yolo_cropped_images(image)

    # Instance-Aware Retrieval Distributor
    top8 = []
    already_suggested_names = set()
    other_possibilities = []
    base, rem = divmod(8, len(cropped_images)) 
    for i, img in enumerate(cropped_images):
        k = base + (1 if i < rem else 0)
        clipmlp_category, clip_feat = my_clipmlp.classify_image_clipmlp(image)
        j = 0
        for possibility in my_clipmlp.clip_find_top_k_similar_in_category(clipmlp_category, clip_feat, None, None):
            if j < k:
                name, sim, feat = possibility
                if name not in already_suggested_names:
                    already_suggested_names.add(name)
                    top8.append(possibility)
            else:
                other_possibilities.append(possibility)
            j += 1
            if j == 8:
                break
        for possibility in other_possibilities:
            name, sim, feat = possibility
            if name not in already_suggested_names:
                already_suggested_names.add(name)
                top8.append(possibility)
            j += 1
            if j == 8:
                break


    # Contrastive Re-Ranking to order best fit data according to the LLM's attribute understanding
    myLLM.create_new_chat()
    llm_response = myLLM.query_chat(beginning_llm_prompt, image)
    top8 = my_clipmlp.contrastive_reranking(top8, llm_response)

    prods = []
    for rank, (p, sim, feat) in enumerate(top8, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[clipmlp_category.replace('\\','_')][product_file_name][1]}"
        prods.append(Product(Image.open(f"shoes\\{clipmlp_category}\\{product_file_name}"), product_price))  
    
    myLLM.clipcategory = clipmlp_category
    myLLM.last_clip_feat = clip_feat
    myLLM.last_pricerange = None
    myLLM.last_brand = None
    myLLM.retrieved_feats = [tup[2] for tup in top8]

    return prods, llm_response


def refine_recommendations(msg: str, hist: List[Tuple[str, str]], prods: List[Product]) -> Tuple[List[Product], str]:
    if myLLM.chat_msg_count == 1:
        llm_response = myLLM.query_chat(second_llm_prompt + msg + clip_category_prompt + myLLM.clipcategory.replace('\\','_')
        + clip_preferences_prompt + myLLM.preferences)
    else:
        llm_response = myLLM.query_chat(msg + '\n' + clip_category_prompt + myLLM.clipcategory.replace('\\','_') 
        + clip_preferences_prompt + myLLM.preferences)

    category, query, pricerange, brand, preferences, msg = myLLM.extract_data_from_followup_responses(llm_response)
    print('-------Data retrieved from user msg-------')
    print(category)
    print(query)
    print(pricerange)
    print(brand)
    print(myLLM.preferences + ' ' + preferences)
    print('--------------')
    feat = my_clipmlp.encode_one_text(query)
    top8 = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand)
    myLLM.clipcategory = category
    myLLM.last_clip_feat = feat
    myLLM.last_pricerange = pricerange
    myLLM.last_brand = brand
    myLLM.preferences = myLLM.preferences + ' ' + preferences
    myLLM.retrieved_feats = [tup[2] for tup in top8]

    prods = []
    for rank, (p, sim, feat) in enumerate(top8, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price))

    if prods == []:
        msg = "I'm sorry, we don't have any available stock of that particular brand at your price range. Please try something else."
    
    return prods, msg

def recommend_less(idx, prods):
    prod_selected_name = prods[idx].name
    myLLM.last_clip_feat -= 0.3*myLLM.retrieved_feats[idx]
    category = myLLM.clipcategory
    feat = myLLM.last_clip_feat
    pricerange = myLLM.last_pricerange
    brand = myLLM.last_brand 
    top8 = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand)
    prods = []
    for rank, (p, sim, feat) in enumerate(top8, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price))
    
    return prods, "Got it! We'll tune our recommendations accordingly!"



# -----------------------------------------------------------------------------
#  UI
# -----------------------------------------------------------------------------

def launch_app():
    CSS = """
    html, body, .gr-app {height:100%; margin:0}
    .top-row {height:100%; align-items:stretch;}      /* columns share height */
    .gr-block {margin-bottom:0}
    #rhs_col {display:flex; flex-direction:column; flex:1 1 auto;}
    #product_gallery {flex:1 1 auto; min-height:0; overflow-y:auto; max-height: 500px;}
    """

    with gr.Blocks(css=CSS) as demo:
        # ------------------------------------------------------------------
        # Persistent UI state
        # ------------------------------------------------------------------
        chat_hist      = gr.State([])   # [(user_msg, bot_msg), …]
        prod_state     = gr.State([])   # [Product, …] shown in gallery
        sel_idx_state  = gr.State(-1)   # index of item in detail view (–1 = none)

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        with gr.Row(elem_classes=["top-row"]):
            # -------------------- LEFT COLUMN ----------------------------
            with gr.Column(scale=3):
                gr.Markdown("### 1️⃣  Upload")
                img_in  = gr.Image(type="filepath", label="Inspiration")
                sub_btn = gr.Button("Get recommendations", variant="primary")

                gr.Markdown("### 2️⃣  Chat")
                chat     = gr.Chatbot()
                user_txt = gr.Textbox(label="Your message")

            # -------------------- RIGHT COLUMN ---------------------------
            with gr.Column(scale=2, elem_id="rhs_col"):
                gr.Markdown("### Recommendations")
                gallery = gr.Gallery(
                    columns=[4], allow_preview=False, elem_id="product_gallery",
                    height="100%", show_label=False, object_fit="cover"
                )

                # ---- Detail pane ---------------------------------------
                with gr.Group(visible=False) as detail_box:
                    gr.Markdown("#### Product details")
                    d_img   = gr.Image()
                    d_price = gr.HTML()

                    # Buttons side-by-side
                    with gr.Row():
                        buy_btn  = gr.Button("Buy now 💳", variant="primary", scale=1)
                        less_btn = gr.Button("Recommend less 👎", variant="secondary", scale=1)

        # ------------------------------------------------------------------
        #  Callbacks
        # ------------------------------------------------------------------
        # ---- first submission (image upload) -----------------------------
        def _submit(img, hist):
            prods, reply = generate_initial_recommendations(img)
            hist = hist or []
            hist.append(("", reply))                      # first bot turn
            return (hist,                                # chat
                    [p.image for p in prods],            # gallery imgs
                    hist,                                # chat state
                    prods,                               # product state
                    -1,                                  # reset selected idx
                    gr.update(visible=False))            # hide detail

        sub_btn.click(
            _submit,
            inputs=[img_in, chat_hist],
            outputs=[chat, gallery, chat_hist, prod_state, sel_idx_state, detail_box],
        )

        # ---- conversational refinement -----------------------------------
        def _chat(msg, hist, prods):
            if not msg:
                return (gr.update(), gr.update(), hist, prods, gr.update(), -1, gr.update())

            new_prods, reply = refine_recommendations(msg, hist, prods)
            hist.append((msg, reply))
            return (hist, [p.image for p in new_prods], hist, new_prods, "", -1, gr.update(visible=False))

        user_txt.submit(
            _chat,
            inputs=[user_txt, chat_hist, prod_state],
            outputs=[chat, gallery, chat_hist, prod_state, user_txt, sel_idx_state, detail_box],
        )

        # ---- show detail when gallery item clicked -----------------------
        def _show(evt: gr.SelectData, prods):
            idx = evt.index if evt else None
            if idx is None or idx >= len(prods):
                return gr.update(), gr.update(), gr.update(visible=False), -1

            p = prods[idx]
            return p.image, f"<h3>Price: {p.price}</h3>", gr.update(visible=True), idx

        gallery.select(
            _show,
            inputs=[prod_state],
            outputs=[d_img, d_price, detail_box, sel_idx_state],
        )

        # ---- “Recommend less” negative-feedback button -------------------
        def _recommend_less(idx, hist, prods):
            # Guard: no item selected
            if idx is None or idx < 0 or idx >= len(prods):
                return gr.update(), hist, prods, gr.update()

            new_prods, bot_reply = recommend_less(idx, prods)

            hist.append((f"Recommend item {idx+1} less", bot_reply))
            return (hist, [p.image for p in new_prods], hist, new_prods, gr.update(visible=False))

        less_btn.click(
            _recommend_less,
            inputs=[sel_idx_state, chat_hist, prod_state],
            outputs=[chat, gallery, chat_hist, prod_state, detail_box],
            )

    demo.queue().launch()



if __name__ == "__main__":
    launch_app()