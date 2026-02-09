import gradio as gr
from search import init, load_indices, encode_text, search_image_only, search_hybrid

init(device="auto")       
load_indices()           

def run_search(query: str, deep: bool, topk: int = 10, alpha: float = 0.5):
    if not query or not query.strip():
        return [], "⚠️ Please enter a query."
    qv = encode_text(query.strip())

    results = search_hybrid(qv, topk=topk, alpha=alpha) if deep \
              else search_image_only(qv, topk=topk)

    gallery = []
    lines = []
    for r in results:
        cap = r["caption"] or ""
        sc  = r["score"]
        gallery.append((r["path"], f"{cap}\n\nscore: {sc:.3f}"))
        lines.append(f"{sc:.3f} | {r['path']}")

    return gallery, "\n".join(lines) if lines else "No results."

def toggle_sliders(checked):
    return (
        gr.update(visible=checked, interactive=checked),
        gr.update(visible=checked, interactive=checked)
    )

def toggle_debug_menu(checked):
    return gr.update(visible=checked, interactive=checked)

with gr.Blocks(title="ImgSearchTR") as demo:
    gr.Markdown("## ImgSearchTR")
    with gr.Row():
      with gr.Column(scale=1):
        query = gr.Textbox(label="Describe your image", placeholder="e.g. uzun bir sokak")
        
        enable_deep  = gr.Checkbox(label="Deep Search", value=False)
        topk  = gr.Slider(1, 30, value=10, step=1, label="Number of Results", visible=False)
        alpha = gr.Slider(0.0, 1.0, value=0.6, step=0.05, visible=False,
                          label="Image-Caption Balance", 
                          info="0 = only caption similarity, 1 = only image similarity")
        enable_deep.change(toggle_sliders, inputs=enable_deep, outputs=[topk, alpha])
        
        go = gr.Button("Search")
        
        debug_menu = gr.Checkbox(label="Debugging", value=False)
        log = gr.Textbox(label="Similarity Scores", lines=6, visible=False)
        debug_menu.change(toggle_debug_menu, inputs=debug_menu, outputs=log)
        
      with gr.Column(scale=2):
        gallery = gr.Gallery(label="Results", columns=4, height="800", show_label=True)

    go.click(run_search, inputs=[query, enable_deep, topk, alpha], outputs=[gallery, log])

if __name__ == "__main__":
    demo.launch(share=True)