# External Packages:
from __future__ import annotations
import gradio as gr
import os
import numpy as np
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

N = 100
p = 0.3

# -----------------------------------------------------------------------------
# Core Backend Calling
# -----------------------------------------------------------------------------

def generate_initial_recommendations(img_path: str | None, n = 8) -> Tuple[List[Product], str]:
    image = Image.open(img_path).copy()
    cropped_images = yolo_shoe_detection.get_yolo_cropped_images(image)

    # Instance-Aware Retrieval Distributor
    topn = []
    already_suggested_names = set()
    other_possibilities = []

    base, rem = divmod(n, len(cropped_images)) 

    for i, img in enumerate(cropped_images):
        k = base + (1 if i < rem else 0)
        clipmlp_category, clip_feat = my_clipmlp.classify_image_clipmlp(img)

        j = 0
        for possibility in my_clipmlp.clip_find_top_k_similar_in_category(clipmlp_category, clip_feat, None, None, N):
            if j < k:
                name, sim, feat = possibility
                if name not in already_suggested_names:
                    already_suggested_names.add(name)

                    # Merge: build Product here
                    product_file_name = os.path.basename(name)[:-4] + '.jpg'
                    product_image_path = f"shoes\\{clipmlp_category}\\{product_file_name}"
                    product_image = Image.open(product_image_path)
                    product_price = f"₹{my_clipmlp.brand_mapping[clipmlp_category.replace('\\','_')][product_file_name][1]}"
                    product = Product(product_image, product_price)

                    topn.append((product, sim, feat))  # Save product instead of filename
            else:
                other_possibilities.append(possibility)
            j += 1
            if j == n:
                break

        for possibility in other_possibilities:
            name, sim, feat = possibility
            if name not in already_suggested_names:
                already_suggested_names.add(name)

                # Merge: build Product here
                product_file_name = os.path.basename(name)[:-4] + '.jpg'
                product_image_path = f"shoes\\{clipmlp_category}\\{product_file_name}"
                product_image = Image.open(product_image_path)
                product_price = f"₹{my_clipmlp.brand_mapping[clipmlp_category.replace('\\','_')][product_file_name][1]}"
                product = Product(product_image, product_price)

                topn.append((product, sim, feat))
            j += 1
            if j == n:
                break

    # Contrastive Re-Ranking using LLM
    myLLM.create_new_chat()
    llm_response = myLLM.query_chat(beginning_llm_prompt, image)
    topn = my_clipmlp.contrastive_reranking(topn, llm_response)

    prods = [tup[0] for tup in topn]  # Now we only extract the Product from each tuple

    # LLM context update
    myLLM.clipcategory, myLLM.last_clip_feat = my_clipmlp.classify_image_clipmlp(image)
    myLLM.last_pricerange = None
    myLLM.last_brand = None
    myLLM.retrieved_feats = [tup[2] for tup in topn]

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
    print(preferences)
    print('--------------')
    feat = my_clipmlp.encode_one_text(query)
    topn = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand, N)
    myLLM.clipcategory = category
    myLLM.last_clip_feat = feat
    myLLM.last_pricerange = pricerange
    myLLM.last_brand = brand
    myLLM.preferences = preferences
    myLLM.retrieved_feats = [tup[2] for tup in topn]

    prods = []
    for rank, (p, sim, feat) in enumerate(topn, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price))

    if prods == []:
        msg = "I'm sorry, we don't have any available stock of that particular brand at your price range. Please try something else."
    
    return prods, msg

def recommend_less(idx, prods):
    prod_selected_name = prods[idx].name
    myLLM.last_clip_feat = (1-p)*np.array(myLLM.last_clip_feat) + p*np.array(myLLM.retrieved_feats[idx])
    category = myLLM.clipcategory
    feat = myLLM.last_clip_feat
    pricerange = myLLM.last_pricerange
    brand = myLLM.last_brand 
    topn = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand, k=N)
    prods = []
    for rank, (p, sim, feat) in enumerate(topn, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price))
    
    return prods, "Got it! We'll tune our recommendations accordingly!"

def recommend_more(idx, prods):
    prod_selected_name = prods[idx].name
    myLLM.last_clip_feat = (1+p)*np.array(myLLM.last_clip_feat) - p*np.array(myLLM.retrieved_feats[idx])
    category = myLLM.clipcategory
    feat = myLLM.last_clip_feat
    pricerange = myLLM.last_pricerange
    brand = myLLM.last_brand 
    topn = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand, k=N)
    prods = []
    for rank, (p, sim, feat) in enumerate(topn, start=1):
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
.top-row {height:100%; align-items:stretch;}
.gr-block {margin-bottom:0}
#rhs_col {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
}
#scroll_wrapper {
    flex: 1 1 auto;
    height: 200px;
    overflow-y: auto;
    border: 1px solid #ccc;
    padding-right: 8px;
}
#product_gallery {
    height: auto !important;
}
"""

    with gr.Blocks(css=CSS) as demo:
        # ------------------------------------------------------------------
        # Persistent UI state
        # ------------------------------------------------------------------
        chat_hist       = gr.State([])   # [(user_msg, bot_msg), …]
        prod_state      = gr.State([])   # [Product, …] shown in gallery
        sel_idx_state   = gr.State(-1)   # index of item in detail view (–1 = none)
        precomputed     = gr.State([])   # Precompute for load_more

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
                chat     = gr.Chatbot(type='tuples')
                user_txt = gr.Textbox(label="Your message")

            # -------------------- RIGHT COLUMN ---------------------------
            with gr.Column(scale=2, elem_id="rhs_col"):
                gr.Markdown("### Recommendations")
                with gr.Group(elem_id="scroll_wrapper"):   # <== Add wrapper
                    gallery = gr.Gallery(
                        columns=[4],
                        allow_preview=False,
                        elem_id="product_gallery",
                        show_label=False,
                        object_fit="cover"
                    )
                load_more_btn = gr.Button("Load more 🔄", variant="secondary")

                # ---- Detail pane ---------------------------------------
                with gr.Group(visible=False) as detail_box:
                    gr.Markdown("#### Product details")
                    d_img   = gr.Image()
                    d_price = gr.HTML()

                    # Buttons side-by-side
                    with gr.Row():
                        buy_btn  = gr.Button("Buy now 💳", variant="primary", scale=1)
                        less_btn = gr.Button("Recommend less 👎", variant="secondary", scale=1)
                        more_btn = gr.Button("Recommend more", variant="secondary", scale=1)

        # ------------------------------------------------------------------
        #  Callbacks
        # ------------------------------------------------------------------
        # ---- first submission (image upload) -----------------------------
        def _submit(img, hist):
            prods, reply = generate_initial_recommendations(img, n=N)
            hist = hist or []
            hist.append(("", reply))                     # first bot turn
            return (hist,                                # chat
                    [p.image for p in prods[:8]],        # gallery imgs
                    hist,                                # chat state
                    prods[:8],                           # product state
                    -1,                                  # reset selected idx
                    gr.update(visible=False),            # hide detail
                    prods)                               # Store in precomputed

        sub_btn.click(
            _submit,
            inputs=[img_in, chat_hist],
            outputs=[chat, gallery, chat_hist, prod_state, sel_idx_state, detail_box, precomputed],
        )

        # ---- conversational refinement -----------------------------------
        def _chat(msg, hist, prods):
            if not msg:
                return (gr.update(), gr.update(), hist, prods, gr.update(), -1, gr.update())

            new_prods, reply = refine_recommendations(msg, hist, prods)
            hist.append((msg, reply))
            return (hist, [p.image for p in new_prods[:len(prods)]], hist, new_prods[:len(prods)], "", -1, gr.update(visible=False), new_prods)

        user_txt.submit(
            _chat,
            inputs=[user_txt, chat_hist, prod_state],
            outputs=[chat, gallery, chat_hist, prod_state, user_txt, sel_idx_state, detail_box, precomputed],
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
            return (hist, [p.image for p in new_prods[:len(prods)]], hist, new_prods[:len(prods)], gr.update(visible=False), new_prods)

        less_btn.click(
            _recommend_less,
            inputs=[sel_idx_state, chat_hist, prod_state],
            outputs=[chat, gallery, chat_hist, prod_state, detail_box, precomputed],
            )

        # ---- “Recommend more” positive-feedback button -------------------
        def _recommend_more(idx, hist, prods):
            # Guard: no item selected
            if idx is None or idx < 0 or idx >= len(prods):
                return gr.update(), hist, prods, gr.update()

            new_prods, bot_reply = recommend_more(idx, prods)

            hist.append((f"Recommend item {idx+1} more", bot_reply))
            return (hist, [p.image for p in new_prods[:len(prods)]], hist, new_prods[:len(prods)], gr.update(visible=False), new_prods)

        more_btn.click(
            _recommend_more,
            inputs=[sel_idx_state, chat_hist, prod_state],
            outputs=[chat, gallery, chat_hist, prod_state, detail_box, precomputed],
        )

        # ---- Generate more images similar to the prompt ------------------
        def _load_more(precomputed, prods, hist):
            if len(prods) == 0:
                hist.append(("Load more similar items", "Please upload an inspiration image first"))
                return hist, [p.image for p in precomputed], hist, precomputed
            
            if len(prods) + 8 > len(precomputed):
                hist.append(("Load more similar items", "Sorry, there are no more items available"))
                return hist, [p.image for p in precomputed], hist, precomputed
            
            hist.append(("Load more similar items", "Okay, loading 8 new items!!"))
            return hist, [p.image for p in precomputed[:len(prods)+8]], hist, precomputed[:len(prods)+8]

        load_more_btn.click(
            _load_more,
            inputs=[precomputed, prod_state, chat_hist],
            outputs=[chat, gallery, chat_hist, prod_state]
        )

    demo.queue().launch()


if __name__ == "__main__":
    launch_app()