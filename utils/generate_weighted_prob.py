# Script for: a*edit_distace + b*phrase_alignments = c
w_A = 0
w_B = 1
path_A = 'alignments/phrase_alignments.txt' # source | target | align_prob
path_B = 'alignments/levenstein_ratio_alignments.txt'  # source | target | align_prob
path_C = 'alignments/combined_alignments_p'+str(w_A)+'_e'+str(w_B)+'.txt'  # source | target | A | B | align_prob


def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)

def normalizeDist(prob_dict):
    for key in prob_dict:
        #normed = normalize(raw)
        raw = [prob_dict[key][y] for y in prob_dict[key]]
        normed = normalize(raw)
        i = 0
        for y in prob_dict[key]:
            prob_dict[key][y] = float(normed[i])
            i+=1
    return prob_dict

def GenProb(path, normalize):
    prob_dict  = {}
    with open(path, 'r') as af:
        #with open(path_B, 'r') as bf:
        for lines in af:
            source = lines.split()[0]
            target = lines.split()[1]
            prob = lines.split()[-1]
            if source not in prob_dict:
                prob_dict[source] = {}
            if target not in prob_dict[source]:
                prob_dict[source][target] = float(prob)
    if normalize:
        prob_dict = normalizeDist(prob_dict)
    return prob_dict


def CombineProb(weight_A, weight_B, prob_dict_A, prob_dict_B):
    prob_dict_C = {}
    for source in prob_dict_A:
        if source in prob_dict_B:
            for target in prob_dict_A[source]:
                if target in prob_dict_B[source]:
                    if source not in prob_dict_C:
                        prob_dict_C[source] = {}
                    if target not in prob_dict_C[source]:
                        prob_dict_C[source][target] = weight_A*float(prob_dict_A[source][target]) \
                                                    + weight_B*float(prob_dict_A[source][target])
                # if target is missing in other
                else:
                    if source not in prob_dict_C:
                        prob_dict_C[source] = {}
                    if target not in prob_dict_C[source]:
                        prob_dict_C[source][target] = weight_A*float(prob_dict_A[source][target])

    return prob_dict_C

def PrintProb(prob_dict_C, path_C, prob_dict_A, prob_dict_B):
    with open(path_C, 'w') as wp:
        for key in prob_dict_C:
            #normed = normalize(raw)
            raw = [prob_dict_C[key][y] for y in prob_dict_C[key]]
            normed = normalize(raw)
            i = 0
            for y in prob_dict_C[key]:
                A,B = 0, 0
                if key in prob_dict_A and y in prob_dict_A[key]:
                    A = prob_dict_A[key][y]
                if key in prob_dict_B and y in prob_dict_B[key]:
                    B = prob_dict_B[key][y]
                wp.write(key+' '+y+' '+str(A)+' '+str(B)+' '+str(raw[i])+' '+str(normed[i])+'\n')
                i+=1



prob_dict_A = GenProb(path_A, normalize=False) # get the distribution in dict
print('Generated distribution A')
prob_dict_B = GenProb(path_B, normalize=True) # get the distribution in dict
print('Generated distribution B')
prob_dict_C = CombineProb(w_A, w_B, prob_dict_A, prob_dict_B) # combine the distribution in dict
print('Generated distribution C')
PrintProb(prob_dict_C, path_C, prob_dict_A, prob_dict_B) # Normalize the prob and print in file
print('Wrote distribution C')
