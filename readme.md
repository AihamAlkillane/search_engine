# Information Retrieval (Search Enginge Project)

## Installation

1. run `pip -r install requirements`
2. For offline training : run `./offline/app.py` and use the routes
   1. `/loader`
   2. `/preprocess`
   3. `/vectorizer`
   4. `/inverted_index`
   5. `/bert_sentence`
   6. `/flat_ip_index`
3. For testing upload the dataset using the `/loader` route to be parsed
4. You can train your models and create indexes using the rest of the Api's
5. For Online : You need to run `./online/app.py` you can use the `/search` route as a POST request to search within the dataset , The form data looks like this :

```JSON
{
    "query":"A dam could make the Congo more usable  While the Congo is mostly navigable it is only usable internally. The rapids cut the middle Congational goods to be easily transported to and from the interior. This would help integrate central Africa economically into the global economy making the region much more attractive for investment.",
    "model_type":"tf-idf",
    "index_type":"inverted_index"
}
```
