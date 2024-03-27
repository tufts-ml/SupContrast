There are two possible ways to write the SupCon loss:

We chose a different way to write the SupCon loss so that differences with our proposed loss would be more clear. However, the two forms of SupCon are equivalent, as we explain below.

Here, we provide a side-by-side comparison of the original SupCon loss and our formulation of it: 

$$
\sum_i \sum_{p} L^{theirs}( i, p), ~~L^{theirs}( i, p) &= \frac{-1}{|\mathcal{P}(i)|} \log \frac{ e^{z_i \cdot z_p} }{ e^{z_i \cdot z_p} + \sum_{q \in \mathcal{P}(i) \setminus p} e^{z_i \cdot z_q} + \sum_{n \in \mathcal{N}(i)} e^{z_i \cdot z_n} }
\\
\sum_S \sum_{p} L^{ours}( S, p), ~~L^{ours}( S, p) &= \frac{-1}{|\mathcal{P}(S)|} \log \frac{ e^{z_p \cdot z_S} }{ e^{z_p \cdot z_S} + \sum_{q \in \mathcal{P}(S) \setminus p} e^{z_p \cdot z_q} + \sum_{n \in \mathcal{N}(S)} e^{z_p \cdot z_n} }
$$


Consider the top line as a translation of Khosla et al’s Eq 2 into our notation. Consider the bottom line as equivalent to our submission’s Eq 3, with minor adjustments for clarity (e.g. written as a function of two indices of the current batch). For notational simplicity, both lines above assume a fixed temperature of $$\tau = 1$$.

As a reminder of notation:
the target integer index is “S” in our notation (“i” in Khosla et al), and ranges from {1, 2, … batch_size}
the partner integer index “p” has the same meaning in both lines: another image in the batch with the same label as the target index. The set of possible partners of the same class for index $$S$$ is denoted as $$\mathcal{P}(S)$$
If (S,p) is a valid input to either loss above, we can guarantee that (p, S) is also a valid input, because p is an index in the current batch, and S is another image in the batch that shares the same class label.

At a quick glance, the two formulations do look distinct, as the top one uses the target index “i” in all terms in the denominator, while the bottom uses the partner index “p”.


However, what really matters is whether the total sum over all pairs of target and partner index is the same. In fact, the total sum of both forms above can be shown to be exactly equivalent. For every $$S,p$$ index pair aggregated in our version (the bottom sum), there is an equivalent term in the top sum, because $$L^{ours}(S,p) = L^{theirs}(p, S)$$, a fact easily verified by inspection.


# Script to verify the two forms are equivalent

We have included the script `check_two_forms_of_supcon.py` in our anonymous code repo. This script shows that for a concrete minibatch of embeddings, the two versions of the loss deliver the same numerical answer.

Here’s an example output for a batch of 6 synthetically generated embedding vectors, where the first 3 indices belong to class0 and the last 3 indices belong to class1. The same total loss value is produced by the two formulations, “ours” and “theirs”:

S= 0 p= 1  | L^ours(S,p) = -1.1122  | L^theirs(p, S) =-1.1122
S= 0 p= 2  | L^ours(S,p) = -0.0364  | L^theirs(p, S) =-0.0364
S= 1 p= 0  | L^ours(S,p) = -0.8681  | L^theirs(p, S) =-0.8681
S= 1 p= 2  | L^ours(S,p) = -2.0168  | L^theirs(p, S) =-2.0168
S= 2 p= 0  | L^ours(S,p) = -0.1271  | L^theirs(p, S) =-0.1271
S= 2 p= 1  | L^ours(S,p) = -2.3515  | L^theirs(p, S) =-2.3515
S= 3 p= 4  | L^ours(S,p) = -1.0308  | L^theirs(p, S) =-1.0308
S= 3 p= 5  | L^ours(S,p) = -0.5278  | L^theirs(p, S) =-0.5278
S= 4 p= 3  | L^ours(S,p) = -2.5750  | L^theirs(p, S) =-2.5750
S= 4 p= 5  | L^ours(S,p) = -1.2277  | L^theirs(p, S) =-1.2277
S= 5 p= 3  | L^ours(S,p) = -1.2569  | L^theirs(p, S) =-1.2569
S= 5 p= 4  | L^ours(S,p) = -0.4125  | L^theirs(p, S) =-0.4125

Total sum over all (S,p) pairs:
 -13.5427 ours
 -13.5427 theirs

