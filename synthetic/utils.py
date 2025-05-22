import pandas as pd
def write_manifest(write_jsons,save_dir,max_seq_length):
    df=pd.DataFrame(write_jsons)
    df['tag']=f'{max_seq_length//1024}k'
    # save_file = save_dir / f'{max_seq_length//1024}k.jsonl'
    # df.to_json(save_file, orient="records", lines=True, force_ascii=True)
    save_file = save_dir / f'{max_seq_length//1024}k.parquet'
    df.to_parquet(save_file)