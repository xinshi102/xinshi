import os
import pandas as pd

def merge_csv_files():
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建5y文件夹的路径
    data_dir = os.path.join(current_dir, 'data', '5y')
    
    # 检查文件夹是否存在
    if not os.path.exists(data_dir):
        print(f"错误：找不到文件夹 {data_dir}")
        return
    
    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(csv_files) != 2:
        print(f"错误：5y文件夹中应该有2个CSV文件，但找到了{len(csv_files)}个")
        return
    
    # 读取并合并CSV文件
    dfs = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # 合并数据框
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 确保时间列是datetime类型
    merged_df['time'] = pd.to_datetime(merged_df['time'])
    
    # 按时间排序
    merged_df = merged_df.sort_values('time')
    
    # 删除重复的时间戳
    merged_df = merged_df.drop_duplicates(subset=['time'])
    
    # 创建输出目录
    output_dir = os.path.join(current_dir, 'data', 'data_w')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并后的文件
    output_path = os.path.join(output_dir, '2y.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成！文件已保存到: {output_path}")
    print(f"合并后的数据包含 {len(merged_df)} 行")

if __name__ == '__main__':
    merge_csv_files() 