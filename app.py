# External Packages:
from __future__ import annotations
import gradio as gr
import os
from dataclasses import dataclass
from typing import List, Tuple, Union
from PIL import Image

# Importing from our various other files:
import my_clipmlp
import llm
myLLM = llm.MyLLMClass()

LocalImg = Union[str, Image.Image]

@dataclass
class Product:
    image: LocalImg
    price: str
    name: str = ""

beginning_llm_prompt = """You are a friendly personal shopping assistant. You have just been shown an image of a product the user likes. 

1. Start by briefly describing what you see in the image (e.g. category, color, key style elements).  
2. Then invite the user to share any extra preferences or context so you can refine your suggestions.  
3. Ask about things like their preferred styles, colors, sizes, brands, occasions, or budget limits.  
4. Keep your tone warm, conversational, and helpful—imagine you're a personal stylist in a boutique.  

Example output:
“Nice choice! I see a pair of black leather ankle boots with a pointed toe and block heel. Do you have a favorite brand or heel height? Or perhaps a color or material you’d like to explore next? Let me know your budget and any style details (like casual vs. dressy), and I'll pull up some perfect matches for you!”

"""

second_llm_prompt = """
For the next incoming messages, you have to now follow these instructions:
Your job on every user message:

1. Figure out which ONE of the following category strings best matches what the user now wants.  
(If they haven't clearly switched, keep the category you last output.)
Boots_Ankle
Boots_Knee High
Boots_Mid-Calf
Boots_Over the Knee
Boots_Prewalker Boots
Sandals_Athletic
Sandals_Flat
Sandals_Heel
Shoes_Boat Shoes
Shoes_Clogs and Mules
Shoes_Crib Shoes
Shoes_Firstwalker
Shoes_Flats
Shoes_Heels
Shoes_Loafers
Shoes_Oxfords
Shoes_Prewalker
Shoes_Sneakers and Athletic Shoes
Slippers_Boot
Slippers_Slipper Flats
Slippers_Slipper Heels


2. Write a concise, embed-ready description of the product(s) the user wants (colour, material, heel height, brand, occasion, budget, etc.).

3. **Return exactly two lines**—nothing else, no markdown:
CATEGORY: <one category from the list>
QUERY: <search description in 1-3 sentences>
MESSAGE: <what you, as a friendly personal shopping assistant, would say after presenting the new set of choices. Invite them to ask further follow-up questions.>

Examples (Note, I have not filled in the MESSAGE part. However, you, as a friendly shopping assistant, should fill it in as needed.)
User: “Do you have these boots in tan?” →  
CATEGORY: Boots_Ankle
QUERY: tan leather ankle boots with block heel, under $150
MESSAGE: <insert message>

User: “Actually show me some flat strappy sandals for a beach wedding.” →  
CATEGORY: Sandals_Flat
QUERY: white or nude flat strappy sandals suitable for beach wedding
MESSAGE: <insert message>

Never add other words, greetings, or JSON. Just the two lines.

Your First User Message:
"""

clip_category_prompt = "The last detected category up until this message was "

# -----------------------------------------------------------------------------
# Core Backend Calling
# -----------------------------------------------------------------------------

def generate_initial_recommendations(img_path: str | None) -> Tuple[List[Product], str]:
    image = Image.open(img_path).copy()

    clipmlp_category, clip_feat = my_clipmlp.classify_image_clipmlp(image)
    top8 = my_clipmlp.clip_find_top_k_similar_in_category(f"shoe_features\\{clipmlp_category}", clip_feat)
    prods = []
    for rank, (p, sim) in enumerate(top8, start=1):
        product_price = "₹1,000"
        prods.append(Product(Image.open(f"shoes\\{clipmlp_category}\\{os.path.basename(p)[:-4]}.jpg"), product_price))
    
    myLLM.create_new_chat()
    myLLM.clipcategory = clipmlp_category
    llm_response = myLLM.query_chat(beginning_llm_prompt, image)

    return prods, llm_response


def refine_recommendations(msg: str, hist: List[Tuple[str, str]], prods: List[Product]) -> Tuple[List[Product], str]:
    if myLLM.chat_msg_count == 1:
        llm_response = myLLM.query_chat(second_llm_prompt + msg+clip_category_prompt + myLLM.clipcategory.replace('\\','_'))
    else:
        llm_response = myLLM.query_chat(msg + '\n' + clip_category_prompt + myLLM.clipcategory.replace('\\','_') )

    category, query, msg = myLLM.extract_data_from_followup_responses(llm_response)
    print(category)
    print(query)
    myLLM.clipcategory = category
    feat = my_clipmlp.encode_one_text(query)
    top8 = my_clipmlp.clip_find_top_k_similar_in_category(f"shoe_features\\{category}", feat)
    prods = []
    for rank, (p, sim) in enumerate(top8, start=1):
        product_price = "₹1,000"
        prods.append(Product(Image.open(f"shoes\\{category}\\{os.path.basename(p)[:-4]}.jpg"), product_price))
    
    return prods, msg

# -----------------------------------------------------------------------------
#  UI
# -----------------------------------------------------------------------------

def launch_app():
    CSS = """
    html, body, .gr-app {height:100%; margin:0}
    .top-row {height:100%; align-items:stretch;}  /* columns share height */
    .gr-block {margin-bottom:0}
    #rhs_col {display:flex; flex-direction:column; flex:1 1 auto; }
    #product_gallery {flex:1 1 auto; min-height:0; overflow-y:auto; max-height: 500px;}
    """

    with gr.Blocks(css=CSS) as demo:
        chat_hist = gr.State([])
        prod_state = gr.State([])

        with gr.Row(elem_classes=["top-row"]):
            # Left column ------------------------------------------------------
            with gr.Column(scale=3):
                gr.Markdown("### 1️⃣  Upload")
                img_in = gr.Image(type="filepath", label="Inspiration")
                sub_btn = gr.Button("Get recommendations", variant="primary")

                gr.Markdown("### 2️⃣  Chat")
                chat = gr.Chatbot()
                user_txt = gr.Textbox(label="Your message")

            # Right column -----------------------------------------------------
            with gr.Column(scale=2, elem_id="rhs_col"):
                gr.Markdown("### Recommendations")
                gallery = gr.Gallery(
                    columns=[4], allow_preview=False, elem_id="product_gallery",
                    height="100%", show_label=False, object_fit="cover"
                )

                with gr.Group(visible=False) as detail_box:
                    gr.Markdown("#### Product details")
                    d_img = gr.Image()
                    d_price = gr.HTML()
                    gr.Button("Buy now 💳", variant="primary")

        # ---- Callbacks -------------------------------------------------------
        def _submit(img, hist):
            prods, reply = generate_initial_recommendations(img)
            hist = hist or []
            hist.append(("", reply))                    # first user turn had no text
            return (hist,                               # chat
                    [p.image for p in prods],           # gallery images
                    hist,                               # update hidden state
                    prods,                              # products in state
                    gr.update(visible=False))           # hide details box

        sub_btn.click(
            _submit,
            inputs=[img_in, chat_hist],
            outputs=[chat, gallery, chat_hist, prod_state, detail_box],
        )

        def _chat(msg, hist, prods):
            if not msg:
                # nothing typed ⇒ leave box as-is
                return (gr.update(),              # chat
                        gr.update(),              # gallery
                        hist,                     # state unchanged
                        prods,                    # …
                        gr.update(),              # leave textbox
                        gr.update())              # detail box

            new_p, reply = refine_recommendations(msg, hist, prods)
            hist.append((msg, reply))
            return (hist,                         # updated chat
                    [p.image for p in new_p],     # new gallery imgs
                    hist,                         # store chat state
                    new_p,                        # store product state
                    "",                           # <-- clear textbox here
                    gr.update(visible=False))     # hide detail box


        user_txt.submit(
            _chat,
            inputs=[user_txt, chat_hist, prod_state],
            outputs=[chat, gallery, chat_hist, prod_state, user_txt, detail_box],
        )

        def _show(evt: gr.SelectData, prods: List[Product]):
            idx = evt.index if evt else None
            if idx is None or idx >= len(prods):
                return gr.update(), gr.update(), gr.update(visible=False)
            p = prods[idx]
            return p.image, f"<h3>Price: {p.price}</h3>", gr.update(visible=True)

        gallery.select(_show, inputs=[prod_state], outputs=[d_img, d_price, detail_box])

    demo.queue().launch()


if __name__ == "__main__":
    launch_app()