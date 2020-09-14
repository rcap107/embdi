License
---
```
   Copyright 2020 Riccardo CAPPUZZO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

EmbDI datasets
---
The datasets contained in this directory were used while working with EmbDI
(https://gitlab.eurecom.fr/cappuzzo/embdi) on the relevant paper. Please refer to
the full repository for more info.

What is provided here was sourced mostly from
[The Magellan Data Repository](https://sites.google.com/site/anhaidgroup/useful-stuff/data#TOC-The-Corleone-Data-Sets).
For each dataset, three tables are provided: table-A and table-B
are slightly modified versions of the original tables (lower cased, spaces replaced
by `_`, some special characters removed), while the third table is the concatenation
of tables A and B.

Edgelists
---
Edgelists are the data structures used by EmbDI. They are generated starting from
each concatenated dataset and are then fed to the algorithm.

EQ tests
---
The `EQ tests` folder contains all the tests used to perform the Embeddings Quality
evaluation in the paper.
