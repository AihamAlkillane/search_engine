import ir_datasets

dataset_name = "lotte/writing/dev/search"

# تحميل الداتاست
dataset = ir_datasets.load(dataset_name)

# طباعة بعض الأمثلة على الوثائق
for i, doc in enumerate(dataset.docs_iter()):
    print(doc)
    if i >= 2:  # نطبع أول 3 وثائق فقط كمثال
        break

# طباعة بعض الأمثلة على الاستعلامات (queries)
for i, query in enumerate(dataset.queries_iter()):
    print(query)
    if i >= 2:
        break

# طباعة بعض الأمثلة على الـ qrels (التقييمات)
for i, qrel in enumerate(dataset.qrels_iter()):
    print(qrel)
    if i >= 2:
        break
