
import matplotlib.pyplot as plt

def get_data():
    files = ["article2.output.txt", "articles1.output.txt",  "articles3.output.txt"]
    collector_f = {}
    for f in files:
        collector_f[f] = []
        with open(f, "r") as fh:
            for line in fh:
                parts = line.replace('(', '').replace(')', '').split(",")
                if '[' in parts[0]:
                    continue
                if len(parts) > 1:
                    c = parts[1].replace('\'', '').replace("\n", '')
                    subparts = c.split("+")
                    for sp in subparts:
                        minipart = sp.replace('"','').split("*")
                        clean = [m.rstrip().lstrip() for m in minipart]
                        collector_f[f].append(clean)
    return collector_f


def order_information_and_plot():
    data = get_data()
    box = {}
    for key in data.keys():
        for item in data[key]:
            if item[1] in box:
                box[item[1]] += float(item[0])
            else:
                box[item[1]] = float(item[0])
    counter = 0
    index = []
    label = []
    weight = []
    for k in box.keys():
        index.append(counter)
        label.append(k)
        weight.append(box[k])
        counter += 1

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.barh(index, weight, align="center")
    ax.set_yticks(index)
    ax.set_yticklabels(label, fontsize=5)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('weight')
    ax.set_title('Sentimental Analysis - Topic model')
    plt.savefig("sentimental_analysis_topic_model.eps", format="eps", dpi=1000)


if __name__ == '__main__':
    order_information_and_plot()