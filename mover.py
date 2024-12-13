import os
import shutil

def move_dumpall_files(source_root, target_root):
    # source_rootから再帰的にdumpall.datを検索
    for dirpath, dirnames, filenames in os.walk(source_root):
        if 'dumpall.dat' in filenames:
            # 現在のdirpathのsource_rootからの相対パスを取得
            relative_path = os.path.relpath(dirpath, source_root)
            # target_rootの該当するフォルダパスを生成
            target_path = os.path.join(target_root, relative_path)

            # target_pathが存在しない場合はフォルダを作成
            os.makedirs(target_path, exist_ok=True)
            
            # ファイルを移動（コピー）
            source_file = os.path.join(dirpath, 'dumpall.dat')
            target_file = os.path.join(target_path, 'dumpall.dat')
            shutil.move(source_file, target_file)
            print(f"Moved {source_file} to {target_file}")

# 使用例
source_root = "F:/hata/662_142_136_200split"
target_root = "F:/hata/662_142_136_500split"
move_dumpall_files(source_root, target_root)
