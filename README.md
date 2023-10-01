## Typing Errors in Factual Knowledge Graphs: Severity and Possible Ways Out
[[Paper]](https://arxiv.org/abs/2102.02307)  [[Data]](https://drive.google.com/drive/folders/18PVGaDJy_JaJV8jezYeRYteFlomoBfnt?usp=sharing)

### Quick start
1. Download `data.tar.xz` from [Google Drive](https://drive.google.com/drive/folders/18PVGaDJy_JaJV8jezYeRYteFlomoBfnt?usp=sharing).
2. Extract the data
```bash
mkdir data
tar -C data -xJf data.tar.xz
```
3. Install dependencies `pip3 install -r requirements.txt`
4. Run `python3 create-dataset.py` to create a version of DBpedia-c dataset.
5. Run `python3 model.py model-name [eval]`. (Python >= 3.6)

### Advanced
* The data are available in a more portable json format. You can download `json-version.tar.xz` from [Google Drive](https://drive.google.com/drive/folders/18PVGaDJy_JaJV8jezYeRYteFlomoBfnt?usp=sharing).
* Run `python3 features/{text,property,surface}.py` to pre-train the feature extractors.
* The `entities` table in `corpus.db` (SQLite3) stores `(name, abstract, properties)` tuple from the DBpedia dump.
* The `annotations` table stores annotations made during the active learning process, with `(name, orig_label, gold_label)` as columns.
* The `hypernyms` table stores `(entity, hypernym)` pairs from the LHD dataset.

### Reference
Please cite the following papers if you intend to use the data or code.
```bib
@misc{yao2021typing,
  author       = {Peiran Yao and
                  Denilson Barbosa},
  editor       = {Jure Leskovec and
                  Marko Grobelnik and
                  Marc Najork and
                  Jie Tang and
                  Leila Zia},
  title        = {Typing Errors in Factual Knowledge Graphs: Severity and Possible Ways
                  Out},
  booktitle    = {{WWW} '21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia,
                  April 19-23, 2021},
  pages        = {3305--3313},
  publisher    = {{ACM} / {IW3C2}},
  year         = {2021},
  url          = {https://doi.org/10.1145/3442381.3449977},
  doi          = {10.1145/3442381.3449977},
}

@inproceedings{caminhas2019detecting,
  author       = {Daniel Caminhas and
                  Daniel Cones and
                  Natalie Hervieux and
                  Denilson Barbosa},
  editor       = {Donatella Firmani and
                  Valter Crescenzi and
                  Andrea De Angelis and
                  Xin Luna Dong and
                  Maurizio Mazzei and
                  Paolo Merialdo and
                  Divesh Srivastava},
  title        = {Detecting and Correcting Typing Errors in DBpedia},
  booktitle    = {Proceedings of the 1st International Workshop on Challenges and Experiences
                  from Data Integration to Knowledge Graphs co-located with the 25th
                  {ACM} {SIGKDD} International Conference on Knowledge Discovery {\&}
                  Data Mining {(KDD} 2019), Anchorage, Alaska, August 5, 2019},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {2512},
  publisher    = {CEUR-WS.org},
  year         = {2019},
  url          = {https://ceur-ws.org/Vol-2512/paper6.pdf},
  timestamp    = {Fri, 10 Mar 2023 16:22:46 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/CaminhasCHB19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{lehmann2015dbpedia,
  title     = {DBpedia--a Large-Scale, Multilingual Knowledge Base Extracted from Wikipedia},
  author    = {Lehmann, Jens and Isele, Robert and Jakob, Max and Jentzsch, Anja and Kontokostas, Dimitris and Mendes, Pablo N and Hellmann, Sebastian and Morsey, Mohamed and Van Kleef, Patrick and Auer, S{\"o}ren and others},
  journal   = {Semantic Web},
  volume    = {6},
  number    = {2},
  pages     = {167--195},
  year      = {2015},
  publisher = {IOS Press}
}
```

### License
#### Data

Under [CC BY-SA 3.0](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License) and [GNU FDL](https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_GNU_Free_Documentation_License).

#### Code

MIT License

Copyright (c) 2021 Peiran Yao & Denilson Barbosa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
