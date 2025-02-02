# PPT Template parsing
Read in the ppt file and convert each slide into an image

```python
presentation = Presentation.from_file(
    pjoin(pptx_config.RUN_DIR, "source.pptx"), pptx_config
)

ppt_to_images(pjoin(pptx_config.RUN_DIR, "source.pptx"), ppt_image_folder)
```

Once the slides has been converted into images, a vision language model is invoked to caption the slides
```python
labler = ImageLabler(presentation, pptx_config)
progress.run_stage(labler.caption_images)
#Prompt: see prompts/caption.txt
```

This generates a json containing some information on each slide image and their caption (see runs/pptx/default_template/image_stats.json)

Stage Input:
- pptx file
- VLM

Stage Output:
- Images of each slide
- image_stats.json

Possible errors:
- package failing to parse slides in pptx


# PDF Parsing
Directly parses the pdf file

```python
text_content = progress.run_stage(
    parse_pdf,
    pjoin(RUNS_DIR, "pdf", pdf_md5, "source.pdf"),
    parsedpdf_dir,
    marker_model,
)
```

Stage input:
- pdf file
- pdf markdown conversion model (from python package `marker`)

Stage Output:
- source.md (text only)
- images from the pdf
- meta.json (pdf meta data)

# PDF Image captioning
Caption the image files that were extracted from the pdf via a VLM.

```python
try:
    images[pjoin(parsedpdf_dir, k)] = [
        llms.vision_model(
            caption_prompt, [pjoin(parsedpdf_dir, k)]
        ),
        PIL.Image.open(pjoin(parsedpdf_dir, k)).size,
    ]
except Exception as e:
    logger.error(f"Error captioning image {k}: {e}")

# See prompts/caption.txt for prompt
```
Stage Input:
- images from the pdf
- VLM

Stage Output:
- caption.json

# PDF markdown refine

Convert the parsed source.md file into a json file via a LM. There is implicit summarization and editing.

```python
doc_json = llms.language_model(
    REFINE_TEMPLATE.render(markdown_document=text_content), return_json=True
)

# See prompts/document_refine.txt for prompt
```

Stage Input:
- source.md
- LM

Stage Output:
- refined_doc.json

# Slide Induction
## Template generation
Create a template pptx file that only keeps the layout of parsed source.pptx, and generate images for each slide of the template.

```python
deepcopy(presentation).save(
    pjoin(pptx_config.RUN_DIR, "template.pptx"), layout_only=True
)
ppt_to_images(
    pjoin(pptx_config.RUN_DIR, "template.pptx"),
    pjoin(pptx_config.RUN_DIR, "template_images"),
)
```
Stage Input:
- source.pptx

Stage Output:
- template.pptx
- images of each slide of template.pptx

## Functional Clustering
Split the slides into different categories based on their function via LM
```python
#induct.py:SlideInducter.layout_induct

content_slides_index, functional_cluster = self.category_split()
# See prompts/category_split.txt
```
Stage Input:
- parsed source.pptx object
- LM

Stage Output:
- split_cache.json (see runs/pptx/default_template/template_induct/backend/split_cache.json)

## Layout Clustering

Split the slides into different categories based on their layout. Clusters are generated via image cosine similarity of each slide, cluster description/name is generated via VLM.

```python
#induct.py:SlideInducter.layout_induct

self.layout_split(content_slides_index)
# See prompts/ask_category.txt
```
Stage Input:
- template.pptx slide images
- VLM

Stage Output:
- ?

## Content Induction

Perform content schema extraction for the presentation using a LM.

```python
#induct.py:SlideInducter.content_induct
schema = llms.language_model(
    content_induct_prompt.render(
        slide=self.prs.slides[cluster["template_id"] - 1].to_html(
            element_id=False, paragraph_id=False
        )
    ),
    return_json=True,
)

# See prompts/content_induct.txt
```

Stage Input:
- parsed source.pptx object
- LM

Stage Output:
- induct_cache.json (see runs/pptx/default_template/template_induct/backend/induct_cache.json)

# PPT generation

PPTCrew takes presentation (step 3.), slide_induction (step 5.), and doc_json + images (step 4.) as input and generates a new pptx file.

a). For doc_json (refined_doc.json), it will use it to generate contents for each slide, conditioned by number of slides. (Not conditioned by the layout of the slide). Example output:

```json
    {
    "Opening of the History of OpenAI": {
        "layout": "opening:text",
        "subsections": [],
        "description": "Introduce the topic of OpenAI, highlighting its historical significance and the overarching 
    vision behind its creation."
    },
    "Key Milestones and Impact of OpenAI": {
        "layout": "Bullet Points with Highlighted Key Terms:text",
        "subsections": ["Foundation of OpenAI", "Core Mission and Principles", "Early Research and Strategic Shifts",
    "Breakthroughs in AI Models", "Global Influence in AI Development", "Ongoing Challenges and Future Goals"],
        "description": "Provide a concise overview of the foundational aspects, achievements, and global impact of 
    OpenAI, closing with an emphasis on the challenges ahead."
    }
    }
```


b). After generating the contents of each slide, it will use the layout_induct (step 6.) to generate the contents to match the specific layout of each slide. Prompt in `./roles/planner.yaml`
Example output:

```json
    {
    "main_title": {
        "data": ["History of OpenAI"]
    },
    "subtitle": {
        "data": ["Exploring its origins and visionary impact"]
    },
    "presenter": {
        "data": ["Presenter Name"]
    }
    }
```

c). Then it will call the LLM to generate editing code, and execute the code to generate the slides of the final pptx file. Example editing code:


```python
# ('main_title', 'text', 'quantity_change: 0', ['Tourism & Culture:'], ['History of OpenAI'])
replace_paragraph(0, 0, "History of OpenAI")

# ('subtitle', 'text', 'quantity_change: 0', ['APPRECIATING THE TANGIBLE & THE INTANGIBLE OF BHUBANESWAR THROUGH 
CULTURAL TOURISM'], ['Exploring its origins and visionary impact'])
replace_paragraph(1, 0, "Exploring its origins and visionary impact")

# ('presenter', 'text', 'quantity_change: 0', ['By Ayona Bhaduri'], ['Presenter Name'])
replace_paragraph(1, 1, "Presenter Name")
```


<!-- 
```python
crew = pptgen.PPTCrew(text_model, error_exit=False, retry_times=5)
crew.set_reference(presentation, slide_induction)
crew.generate_pres(generation_config, images, slides_count, doc_json)
```

Stage Input:
-  -->