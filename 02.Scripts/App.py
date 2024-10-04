import pandas as pd
import streamlit as st
import numpy as np
from openai import OpenAI
import pdb
import copy
import json
from scipy.spatial import distance
import time
from PIL import Image
from pathlib import Path


def get_embedding(text, model="text-embedding-3-small"):
    #text = text.replace("\n", " ")
    if pd.isna(text):
        return [0]
    else:
        text = str(text).lower()
        return client.embeddings.create(input = [text], model=model).data[0].embedding


@st.cache_data(ttl=120)
def _ask_chatgpt1(texto, _client, _context='', _model = "gpt-3.5-turbo-0125", _drpta = 'json_object', _temperature=0):
    texto = str(texto)
    response = _client.chat.completions.create(
        model=_model,
        # model="gpt-4o",
        response_format = {"type": _drpta,},
        messages=[
        {"role": "system", "content": _context},
        {"role": "user", "content": texto}],
        seed=12345,
        temperature=_temperature,
        # max_tokens=100
    )
    answer_str = response.dict()['choices'][0]['message']['content']
    return answer_str

@st.cache_data(ttl=120)
def _ask_chatgpt2(texto, _client, _context='', _model = "gpt-4o-mini", _drpta = 'json_object', _temperature=0):
    texto = str(texto)
    response = _client.chat.completions.create(
        model=_model,
        # model="gpt-4o",
        response_format = {"type": _drpta,},
        messages=[
        {"role": "system", "content": _context},
        {"role": "user", "content": texto}],
        seed=12345,
        temperature=_temperature,
        # max_tokens=100
    )
    answer_str = response.dict()['choices'][0]['message']['content']
    return answer_str

def _get_similitud(x, emb):
    return (1-distance.cosine(x, emb))

@st.cache_data(ttl=120)
def _find(str_buscado, dfglos, colname, first=5, return_id = False, id_name = None):
    emb_buscar = get_embedding(str_buscado)
    # calcular distancias/similitud
    # mejor dbscan
    dfglos['similitud'] = dfglos['embedding'].apply(_get_similitud, args=(emb_buscar,))
    dffinded = dfglos.sort_values(['similitud'], ascending=False).iloc[:first,:][colname]#.to_list()
    # debe retornar una lista con los nombres
    return dffinded

path_str = Path(__file__).parent.parent
image = Image.open(path_str / '01.Data/Logo_iaxta.jpeg')
st.image(image, width=200)

st.title("Generador de ideas de aplicación a partir de los proyectos de investigación científica financiados por PROCIENCIA entre 2015 y 2021")
st.header("Elaborado en el marco del concurso #ExprésatePerúConDatos 2024")

key_ = st.secrets["akey"]
client = OpenAI(api_key = key_)
df = pd.read_pickle(path_str / "01.Data/df_embedding.pkl")


if "disabled" not in st.session_state:
    st.session_state.disabled = False

if "messages" not in st.session_state or 0 == len(st.session_state.messages):
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola!, realiza una pregunta o indica el sector en el que se encuentra tu empresa para saber cómo alguno de los 909 proyectos de investigación científica financiados por PROCIENCIA entre 2015 y 2021 puede ayudarte", 'type':'text'}]
    

for j, msg in enumerate(st.session_state.messages):
    largo = len(st.session_state.messages)
    if msg['type']=='text':
        st.chat_message(msg["role"]).write(msg["content"])
    elif msg['type']=='table':
        st.chat_message(msg["role"]).table(msg["content"])
    elif (msg['type']=='pbar') & (j + 1 ==largo):
        with st.chat_message("assistant"):
            progress_text = "Podrás hacer la siguiente pregunta cuando la barra llegue al 100% (60 segundos)"
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.6)
                my_bar.progress(percent_complete + 1, text=progress_text)

def on_submit():
    st.session_state.disabled = True

    st.session_state.messages.append({"role": "user", "content": st.session_state.question, 'type':'text'})
    dfinded = _find(st.session_state.question, df, ['CODIGO_ORDEN','ANIO','ENTIDAD_EJECUTORA_SUBVENCIONADO','DESCRIPCION'])

    prompt1 = """
    ¿Es el input ingresado por el usuario una consulta que requiere una respuesta detallada?. Responde 'Sí' o 'No' en un json con el siguiente formato:
    {'Respuesta': 'Sí' o 'No'}
    """

    json_is_question = _ask_chatgpt1(st.session_state.question, client, _context=prompt1)
    is_question = json.loads(json_is_question)['Respuesta']

    if is_question=='Sí':
        context_2 = "Responde la pregunta del usuario seleccionando los proyectos de investigación de la siguiente lista que pueden resolver su consulta y redacta a detalle como cada proyecto seleccionado le podría ayudar. Si ningún proyecto de investigación de la lista ayuda a responder la pregunta, indica que 'No existe ningún proyecto relaciondo a su consulta'. Si no se puede resolver la consulta con la información, responde 'No cuento con la información para responder esa pregunta'. Proyectos: " + str(dfinded[['ENTIDAD_EJECUTORA_SUBVENCIONADO','DESCRIPCION']].to_dict(orient='records'))
        rpta = _ask_chatgpt2(st.session_state.question, client, _context=context_2, _model='gpt-4o', _drpta='text', _temperature=1)
        st.session_state.messages.append({"role": "assistant", "content": rpta,'type':'text'})
    else:
        st.session_state.messages.append({"role": "assistant", "content": dfinded,'type':'table'})

    #############################
    st.session_state.messages.append({"role": "assistant", "content": "", 'type':'pbar'})

    st.session_state.disabled = False


if question_ := st.chat_input("Ponga su consulta aquí", on_submit=on_submit, disabled=st.session_state.disabled, key="question"):
    pass

