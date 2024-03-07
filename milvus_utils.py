from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)



def connect_milvus(coll_name="face_collection",dim=128):
    fmt = "\n=== {:30} ===\n"


    has = utility.has_collection(coll_name)
    if not has:
        print(f"Does collection face_collection exist in Milvus: False")
        fields = [
            FieldSchema(name="face_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="face_embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "Face Vector Collection")
        print(fmt.format("Create collection face_collection"))
        return Collection(coll_name, schema, consistency_level="Strong")

    else:
        print(f"Does collection face_collection exist in Milvus: {has}")
        return Collection(coll_name)



def extract_features_to_milvus(collection,id,vector):
    collection.insert([id, vector])
    collection.flush()