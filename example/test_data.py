import glob
import os

from tools.robot_data_binding.data_reader.raw_data_load import HDF5Reader

# 取 example/data 下第一个 h5 作为演示(路径相对本文件,便于任意目录运行)
_HERE = os.path.dirname(os.path.abspath(__file__))
H5_PATH = sorted(glob.glob(os.path.join(_HERE, "data", "*", "*.h5")))[0]


def main():
    reader = HDF5Reader("example/data/1094bf5bd16e4426964e1e71f44bd914/s1a3aa2266ae4e568cb903138774a002.h5")
    reader.load()

    # 1) 浏览:列出所有 key + 猜测种类(此时一律 raw、未确认)
    reader.describe()

    # 2) 由使用者标注 key 的语义,再渲染
    # reader.set_kind("metadata.json", "text")
    # reader.render("metadata.json")  # 打印元数据(text 会美化 JSON)

    # 未标注的 key 走 raw 兜底:只给结构 + 首元素预览
    # reader.render("timestamp")

    # 需要图形界面的渲染(先标注种类,再渲染;无头环境会失败,按需打开):
    # reader.set_kind("cameras/head/color/data", "image")
    # reader.render("cameras/head/color/data", frame=0)
    # reader.set_kind("joints/state/arm/position", "series")
    # reader.render("joints/state/arm/position")

    # 4) 或直接交互式浏览:列表 -> 选序号 -> 选可视化方式
    reader.explore()

    reader.close()


if __name__ == "__main__":
    main()
