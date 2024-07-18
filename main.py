from typing import List
# from PyMultiDictionary import MultiDictionary
import hanlp
from hanlp.common.component import Component
from fastapi import FastAPI, Body

tok: Component = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)
pos: Component = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

app = FastAPI()

@app.post("/tok")
def tok_(text: str = Body(...)):
    return tok(text)


@app.post("/pos")
def pos_(text: List[str] = Body(...)):
    return pos(text)
