* `edgelist.py` generates an edgelist given a csv file 

The format of the `edgelist.py` is one of the following:
* `n1,n2,w1,w2` >> node1, node2, weight1to2, weight2to1
* `n1,n2` >> node1, node2, 1, 1
* `n1,n2,w1` >> node1, node2, weight1to2 [no link back]

The first line of the edgelist must contain all the prefixes that define node types (such as RID, CID,
tokens). Prefixes must be written using a specific format:
`[1-7][#-$]__node_type_name`

The first character should be a numeric value in the range `1-7`, which will denote the class of the node type. 
The second character is used to distinguish nodes that contain numeric values (with symbol `#`) from those that contain categorical 
information (`$`). 



The class will influence the behavior 
of the random walks generation algorithm. 

There are 7 possible classes based on the truth table reported below. `first` means that the node may be chosen 
as first value in a sentence. `root` means that the node will be added to the pool of nodes to start from when 
generating sentences. `appears` means that, when the random walks hits the node, the node will appear in the 
sentence, otherwise it will not be saved. 

| Class | First | Root | Appears |
|:-----:|:-----:|:----:|:-------:|
|   0   |   -   |   -  |    -    |
|   1   |   -   |   -  |    +    |
|   2   |   -   |   +  |    -    |
|   3   |   -   |   +  |    +    |
|   4   |   +   |   -  |    -    |
|   5   |   +   |   -  |    +    |
|   6   |   +   |   +  |    -    |
|   7   |   +   |   +  |    +    |




It's possible to flatten all nodes using flatten:all in the configuration file. Alternatively, 
nodes can be flattened based on their type by listing the prefixes to expand separated by a comma. 


TODO:
Add more variations of the prefixes. 
Adding support for compression. 