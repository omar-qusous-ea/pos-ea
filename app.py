# code is adopted from https://huggingface.co/spaces/flair/model_demo/blob/main/app.py and modfied to fit the purpose of this project.
import spacy.displacy
import streamlit as st
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from colorhash import ColorHash

# st.title("Flair NER Demo")
st.set_page_config(layout="centered")

# Block 2: Users can input text
st.subheader("Input your text here")
input_text = st.text_area('Write or Paste Text Below',
                          value='May visited the Eiffel Tower in Paris last May.\n\n'
                                'There she ran across a sign in German that read: "Dirk liebt den Eiffelturm"',
                          height=128,
                          max_chars=None,
                          label_visibility="collapsed")


@st.cache_resource
def get_model():
    return Classifier.load('pos')


def get_html(html: str):
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def color_variant(hex_color, brightness_offset=1):
    """ takes a color like #87c95f and produces a lighter or darker variant
    taken from: https://chase-seibert.github.io/blog/2011/07/29/python-calculate-lighterdarker-rgb-colors.html
    """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x + 2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]  # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])


# Block 3: Output is displayed
button_clicked = st.button("**Click here** to tag the input text", key=None)

if button_clicked:
    splitter = SegtokSentenceSplitter()
    # TODO: perhaps truncate input_text
    sentences = splitter.split(input_text)

    # get the model and predict
    model = get_model()
    model.predict(sentences)

    spacy_display = {"ents": [], "text": input_text, "title": None}

    predicted_labels = set()
    for sentence in sentences:
        for prediction in sentence.get_labels():
            entity_fields = {
                "start": prediction.data_point.start_position + sentence.start_position,
                "end": prediction.data_point.end_position + sentence.start_position,
                "label": prediction.value,
            }

            spacy_display["ents"].append(entity_fields)
            predicted_labels.add(entity_fields["label"])

    # create colors for each label
    colors = {}
    for label in predicted_labels:
        colors[label] = color_variant(ColorHash(label).hex, brightness_offset=85)

    # use displacy to render
    html = spacy.displacy.render(spacy_display,
                                 style="ent",
                                 minify=True,
                                 manual=True,
                                 options={
                                     "colors": colors,
                                 },
                                 )
    style = "<style>mark.entity { display: inline-block }</style>"
    st.subheader("Tagged text")
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)