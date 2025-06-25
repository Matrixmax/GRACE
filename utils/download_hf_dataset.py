from datasets import load_dataset

dataset = load_dataset("tianyang/repobench_java_v1.1",ignore_verifications=True,download_mode='force_redownload')




# dataset = load_dataset("tianyang/repobench_java_v1.1")

# cross_file_first = load_dataset(f"/data/wxl/graphrag4se/GRACE/dataset/hf_datasets/repobench_{language}_v1.1",split=['cross_file_first'])[0]