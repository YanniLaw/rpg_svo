// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/feature_detection.h>
#include <svo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>

namespace svo {
namespace feature_detection {

/**
 * @brief 抽象角点检测器
 *
 * @param img_width 图像宽度
 * @param img_height 图像高度
 * @param cell_size 网格单元大小
 * @param n_pyr_levels 金字塔层数
 */
AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

/**
 * @brief FAST角点检测器
 *
 * @param img_width 图像宽度
 * @param img_height 图像高度
 * @param cell_size 网格单元大小
 * @param n_pyr_levels 金字塔层数
 */
FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

/**
 * @brief 在图像金字塔上做分网格的 FAST 角点检测 + 每格取最优（Shi-Tomasi 打分），把结果包装成 Feature 追加到 fts。
 * 它刻意不做全局 NMS，而是靠网格占用表保证特征在图像上均匀分布、并避免与已有特征重复。
 * 
 * @param frame   // 特征所属帧（构造函数里用它算 bearing vector）
 * @param img_pyr // 图像金字塔
 * @param detection_threshold // Shi-Tomasi 最终评分阈值
 * @param fts     // 输出：追加到已有列表末尾，调用者负责 delete
 */
void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    Features& fts)
{
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    vector<fast::fast_xy> fast_corners;
#if __SSE2__ // X86 SSE2
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners); // 20：FAST 的像素亮度差阈值
#elif HAVE_FAST_NEON // ARM NEON (相同图像在 x86 和 ARM 上检测出的候选角点可能不同)
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else // 通用CPU
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif
    vector<int> scores, nm_corners; // nm_corners 保存的是 fast_corners 中被保留角点的索引,而不是新的角点坐标
    // 给候选角点计算响应分数
    fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
    // 比较局部邻域内的 FAST 分数，抑制较弱的相邻角点
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      // 将该角点映射到网格单元，网格单元的大小是 cell_size_，网格单元的坐标是原始图像坐标系下的坐标
      // 不管候选点来自金字塔哪一层，都会映射到同一个原始图像网格空间
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k]) // 检查网格单元是否已经被占用，如果是，则跳过该角点
        continue;
      // Shi-Tomasi 角点评分
      // 为什么已经进行了 FAST 评分，还需要 Shi-Tomasi？
      // FAST 负责快，Shi-Tomasi 负责质量
      // 因为 FAST 主要用于快速识别亮度变化模式，而 SVO 后面需要通过图像 Patch 进行光度对齐
      // 对于这种优化，具有较好二维梯度结构的角点通常更容易提供稳定的位移约束
      // Shi-Tomasi 正好可以衡量这种性质
      // SVO 使用 Shi-Tomasi 选择角点，实际上是在倾向于选择图像块对齐时具有较好二维可观测性的区域
      // Harris 更偏向用两个特征值构造一个综合角点响应，Shi-Tomasi 则直接考察最弱方向的图像梯度信息。
      const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      // 竞争更新: 每个网格只保留得分最高的角点
      //TODO: 不同金字塔层的 Shi-Tomasi 分数被直接比较，没有显式进行尺度归一化
      if(score > corners.at(k).score)
        // 因为存储的是原始图像坐标，所以要乘 scale  
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }

  // Create feature for every corner that has high enough corner score
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    if(c.score > detection_threshold)
      fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
  });

  resetGrid();
}

} // namespace feature_detection
} // namespace svo

