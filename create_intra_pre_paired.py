import os

def extract_pre_files(input_file_path, output_file_path):
    """
    从文件列表中提取所有术前(pre)文件名称
    """
    pre_files_list = []
    
    # 1. 读取原始文件
    if not os.path.exists(input_file_path):
        print(f"错误: 找不到文件 {input_file_path}")
        return

    print(f"正在读取文件: {input_file_path} ...")
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 过滤提取
    for line in lines:
        # 去除换行符和空格
        clean_name = line.strip()
        
        # 判断是否非空且以指定后缀结尾
        if clean_name and clean_name.endswith('_pre_processed_volume'):
            pre_files_list.append(clean_name)

    # 3. 保存到新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for name in pre_files_list:
            f.write(name + '\n')

    print(f"处理完成！")
    print(f"共扫描到 {len(lines)} 个文件。")
    print(f"提取出 {len(pre_files_list)} 个术前(pre)文件。")
    print(f"新列表已保存至: {output_file_path}")

# --- 执行配置 ---
if __name__ == "__main__":
    # 输入文件名 (您上传的文件名)
    input_txt = "./files_names/all_files.txt" 
    # 输出文件名
    output_txt = "./files_names/train_files_pre_only.txt"
    
    extract_pre_files(input_txt, output_txt)