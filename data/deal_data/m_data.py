import os
import pandas as pd

def merge_csv_files():
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # deal_data文件夹路径就是当前目录
    data_dir = current_dir
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"错误：{data_dir} 文件夹下没有找到任何CSV文件")
        return
    
    # 读取并合并所有CSV文件
    dfs = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 如果有time列，按时间排序并去重
    if 'time' in merged_df.columns:
        merged_df['time'] = pd.to_datetime(merged_df['time'])
        merged_df = merged_df.sort_values('time')
        merged_df = merged_df.drop_duplicates(subset=['time'])
    
    # 保存合并后的文件
    output_path = os.path.join(data_dir, 'merged.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成！文件已保存到: {output_path}")
    print(f"合并后的数据包含 {len(merged_df)} 行")

if __name__ == '__main__':
    merge_csv_files() 