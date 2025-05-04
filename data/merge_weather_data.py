import pandas as pd
import os
from datetime import datetime

def merge_weather_files(input_dir='data', output_file='merged_weather_data.csv'):
    """
    合并data目录下的所有CSV文件
    
    参数:
    input_dir: 输入文件目录
    output_file: 输出文件名
    """
    print("开始合并天气数据文件...")
    
    # 获取目录下所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f != output_file]
    
    if not csv_files:
        print("未找到CSV文件！")
        return
    
    print(f"找到以下CSV文件：")
    for file in csv_files:
        print(f"- {file}")
    
    # 读取并合并所有CSV文件
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(input_dir, file))
            print(f"成功读取文件: {file}, 行数: {len(df)}")
            print(f"列名: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {str(e)}")
    
    if not dfs:
        print("没有成功读取任何文件！")
        return
    
    # 合并所有数据框
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n合并前总行数: {len(merged_df)}")
    print(f"合并后的列名: {list(merged_df.columns)}")
    
    # 数据清理
    # 1. 删除只有时间列的行
    merged_df = merged_df.dropna(subset=[col for col in merged_df.columns if col != 'time'], how='all')
    print(f"删除只有时间列的行后行数: {len(merged_df)}")
    
    # 2. 如果存在时间列，按时间排序
    if 'time' in merged_df.columns:
        merged_df['time'] = pd.to_datetime(merged_df['time'])
        merged_df = merged_df.sort_values('time')
    
    # 3. 删除重复行
    merged_df = merged_df.drop_duplicates()
    print(f"删除重复后行数: {len(merged_df)}")
    
    # 保存合并后的文件
    output_path = os.path.join(input_dir, output_file)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n合并完成！")
    print(f"最终行数: {len(merged_df)}")
    print(f"输出文件: {output_path}")

if __name__ == '__main__':
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'merged_weather_data_{timestamp}.csv'
    
    # 执行合并
    merge_weather_files('data', output_file) 