import pandas as pd
import stanza
from tqdm import tqdm
import script_for_graph
import importlib
from script_for_graph import header_text, tail_text


def norm_form(morph, word):
    return morph.parse(word)[0].normal_form


def load_stop_words():
    stopwords = []
    path_to_file = "stopwords/Stopwords.txt"
    with open(path_to_file, "r", encoding="utf-8") as fl:
        for line in fl:
            stopwords.append(line.strip("\n"))
    return stopwords


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    with open('input_speech.txt') as f:
        all_text = f.read(100000000)
        long_sents = all_text.split(".")
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,ner,depparse')
        triplets = []
        for s in tqdm(long_sents):
            doc = nlp(s)
            for sent in doc.sentences:
                entities = [ent.text for ent in sent.ents]
                res_d = dict()
                temp_d = dict()
                for word in sent.words:
                    temp_d[word.text] = {"head": sent.words[word.head - 1].text, "dep": word.deprel, "id": word.id}
                for k in temp_d.keys():
                    nmod_1 = ""
                    nmod_2 = ""
                    if (temp_d[k]["dep"] in ["nsubj", "nsubj:pass"]) & (k in entities):
                        res_d[k] = {"head": temp_d[k]["head"]}

                        for k_0 in temp_d.keys():
                            if (temp_d[k_0]["dep"] in ["obj", "obl"]) & \
                                    (temp_d[k_0]["head"] == res_d[k]["head"]) & \
                                    (temp_d[k_0]["id"] > temp_d[res_d[k]["head"]]["id"]):
                                res_d[k]["obj"] = k_0
                                break

                        for k_1 in temp_d.keys():
                            if (temp_d[k_1]["head"] == res_d[k]["head"]) & (k_1 == "не"):
                                res_d[k]["head"] = "не " + res_d[k]["head"]

                        if "obj" in res_d[k].keys():
                            for k_4 in temp_d.keys():
                                if (temp_d[k_4]["dep"] == "nmod") & \
                                        (temp_d[k_4]["head"] == res_d[k]["obj"]):
                                    nmod_1 = k_4
                                    break

                            for k_5 in temp_d.keys():
                                if (temp_d[k_5]["dep"] == "nummod") & \
                                        (temp_d[k_5]["head"] == nmod_1):
                                    nmod_2 = k_5
                                    break
                            res_d[k]["obj"] = res_d[k]["obj"] + " " + nmod_2 + " " + nmod_1

                if len(res_d) > 0:
                    triplets.append([s, res_d])

        print(triplets)

        clear_text = lambda x: "".join(i if (i.isdigit()) | (i.isalpha()) | (i in [" "]) else " " for i in x)

        clear_triplets = dict()
        for tr in triplets:
            for k in tr[1].keys():
                if "obj" in tr[1][k].keys():
                    clear_triplets[clear_text(tr[0])] = [k, tr[1][k]['head'], tr[1][k]['obj']]
        for_df = []
        for k in clear_triplets.keys():
            for_df.append([k] + clear_triplets[k])
        df_triplets = pd.DataFrame(for_df, columns=["full_sent", "subject", "verb", "object"])
        df_triplets.shape
        df_triplets["subj_n_f"] = df_triplets["subject"]
        df_triplets["obj_n_f"] = df_triplets["object"]
        df_triplets.head(10)
        df_filtered = df_triplets
        df_filtered.shape
        df_filtered.head(6)
        groups = list(chunks(df_filtered["obj_n_f"].unique(), 100))
        len(groups)
        gr_num = 0
        df_for_draw = df_filtered[df_filtered["obj_n_f"].isin(groups[gr_num])]

        nodes = pd.unique(df_for_draw[["subj_n_f", "obj_n_f"]].values.ravel("K"))

        nodes.shape
        df_d_d = df_for_draw.drop_duplicates(
            subset=["subj_n_f", "obj_n_f", "verb"])[["subj_n_f", "obj_n_f", "verb", "full_sent"]]
        df_d_d.shape, df_for_draw.shape
        info_dict = dict()
        label_dict = dict()
        for cc, raw in enumerate(df_d_d.values):
            info_dict[(raw[0], raw[1])] = {f"sent_{cc}": raw[3]}
            label_dict[(raw[0], raw[1])] = raw[2]

        word_num = dict()
        for c, word in enumerate(nodes):
            word_num[word] = c + 1

        importlib.reload(script_for_graph)

        header_text += """\nvar nodes = new vis.DataSet([\n"""
        for w in nodes:
            header_text += "{"
            header_text += f"""         id: {word_num[w]}, 
                                        label: "{w}"\n"""
            header_text += "},"
        header_text += "   ]);\n"

        header_text += """var edges = new vis.DataSet(["""
        for k in info_dict.keys():
            header_text += "{"
            header_text += f"""       from: {word_num[k[0]]}, 
                            to: {word_num[k[1]]}, 
                            arrows: "to",
                            label: "{label_dict[k]}",
                            info: {info_dict[k]}\n"""
            header_text += "},"
        header_text += "   ]);\n"

        full_text = ""
        full_text += header_text
        full_text += tail_text

        with open(f"Graph_for_group_{gr_num}.html", "w", encoding="utf-8") as f:
            f.write(full_text)