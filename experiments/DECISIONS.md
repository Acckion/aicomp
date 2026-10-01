# 实验决策与待办

目标：当前 phase2 榜单至少 57 分；本地 val400 mAP 不构成目标完成证据。

用户确认：1600 训练样本少于全量 2000；val400 与 phase2 存在分布差异，phase2 更难。

执行顺序：先完成已知轻量改进（评测/TXT 校验、分辨率、NMS/Soft-NMS、切片），再根据证据推进训练改进。不要持续监督训练，最多每 20 分钟检查一次。

## 如何使用验证结果

- 固定 RGB1600 最佳权重，比较同一验证集上的相对增益；绝对 mAP 仅作描述。
- 排名表按 mAP 排序仅便于浏览，不代表自动选定最终方案。
- 同时比较 AP-small、AP-medium、AP-large、AR、各类别变化、推理成本；进一步检查密集场景漏检与重复框。
- 优先保留多个场景/类别收益一致的候选。不能用仅 4 个 tricycle 框、18 个 ball 框造成的波动决定整体方案。
- 密集程度和尺寸子集需按验证标注定义，报告样本量；不将子集 AP 简单平均替代整体 AP。
- 全量模型不在它见过的 val400 上选优或报告独立验证性能。
- 在 2000 张版上迁移候选推理方案；轮次选择参考验证版学习阶段，而非假定两版最优轮次相同。不得简单融合多阶段权重。
- 最终需要正式 phase2 提交反馈，不能以本地超过 55 判定成功。每天两次提交用于有明确差异的候选，不用于大规模扫描。
- 不预设本地增益可等比例迁移到更难的 phase2。测试集不参与训练或人工生成预测。

## 已部署

- scripts/evaluate_variants.py：缓存同一模型的整图/局部推理，离线比较后处理，验证提交 TXT 往返一致性。
- scripts/run_simple_experiments.py：baseline 完成且 GPU 空闲后，先冒烟验证，再用 8 张 GPU 各跑一个推理配置。
- experiments/simple/：状态、原始预测、各类别/尺寸 AP 和 AR、结果排名。
- 当前队列仅评测 RGB1600 对照；后续全量迁移和正式提交仍待结果与提交入口，不宣称已完成。

## 提交工具准备

- evaluate_variants.py 现支持 --predict-only、--annotations、--image-root、--method、--threshold，可生成全量模型 phase2 推理包。
- 预测缓存校验权重与图像清单 SHA256，避免混用不同权重/分辨率结果。
- 提交包逐图校验类别、有限坐标/置信度、100 框上限和文件名唯一性，并保存清单。生成包不等于已提交。
- 已通过 CPU 检查：800 尺寸模型加载 640 训练权重、理想预测 AP=100、类别内 NMS、Soft-NMS、TXT 往返、空预测文件。
- GPU 实际推理与完整 val400 TXT 往返仍由排队的实验进行验证。

## 2026-09-30：裁剪对照结束后的自动流程

- after_targetcrop.py 等待裁剪、普通续训对照均完成，且前一轮 phase2 包已生成；等待只读取完成标记和队列状态。
- 裁剪全量迁移门槛：最后 5 轮总体 AP 中位数比对照至少 +0.1，小目标 AP 至少 +0.25，总体不低于裁剪开训前；验证框数 >=100 的类别没有超过 2 点的中位数下降。是操作筛选，不是显著性检验，也不保证 phase2 收益。
- 通过时，从 ft2000_aug800 的第20轮权重迁移裁剪配置，训练15轮；按照验证裁剪版的最佳训练轮次取全量权重，生成单模型 phase2 包。全量轮次只是学习阶段参照，不在见过的 val400 上选优。
- 独立结构试验 detail800：使用骨干 stride4 浅层特征，通过可训练下采样和零初始化门控接入 stride8；保持三层 decoder 和已有参数形状。不是四层 P2 decoder。与裁剪独立，不把两项同时修改混为一种收益。
- detail800 从相同 ft_aug800 第20轮权重出发，普通续训增强；20轮，三卡单卡 batch1、累积3步，有效 batch9，8.5GiB allocator 上限。新分支 LR1e-4、门控1e-3，旧参数延续较小LR。
- CPU检查已通过：零门控下输出与原 encoder 完全相等；门控与分支梯度有效；配置模型与真实权重兼容；优化器每个训练参数恰好分配一次；迁移筛选拒绝总体/常见类别退化。
- GPU可用后，先进行最大训练尺度的三卡、累积梯度冒烟检查；通过才开训，失败记录原因并停止本队列，不打断其他任务。
- 当前绘图20分钟更新；下一阶段另生成 monitoring/detail 图表与 CSV/PDF。

## 用户最新调整：跳过续训对照，直接优化

- 用户明确要求不做普通续训对照；在对照冒烟阶段停止其控制器、torchrun 和关联工作进程，未完成训练轮次。未停止全量训练。
- after_targetcrop.py --skip-control 立即准备 detail800 三卡训练，不再等待对照或全量提交包。此路线不宣称已做配对归因，也不自动根据缺失对照的裁剪结果作全量推广决定。
- 全量训练的原等待控制器由 package_full_stage.py 接管：已有训练进程保持运行，完成后在 GPU3/4/6 生成包。相同权重、相同推理设置的候选去重，节省推理和提交次数。
- detail800 使用 GPU0/2/5；全量训练及其打包使用另一组卡。monitoring/detail 省略被跳过的对照曲线。

## 细节分支首轮中断与恢复

- 首次正式训练在首轮10个batch之后发生 NCCL collective 次序/尺寸不一致并超时，没有完整 epoch 或权重保存；失败日志和起点指标保留在 experiments/after_targetcrop。
- detail800 改为 find_unused_parameters=True、sync_bn=False；不改变其他已运行训练配置。三卡18 batch回归测试通过，包含rank2单批全部空目标，检验各卡目标数/去噪分支不同情况下的同步。
- 从原始微调权重重新开始，不宣称恢复了未保存的更新。同步超时具体触发点仍需区分条件分支与其他通信因素；回归测试通过不是单独的根因证明。
- scripts/plot_current_training.py 显示当前细节分支完整epoch验证值、轮内累计loss（失败与重启尝试分开），以及全量训练loss/LR/进度；未完成的epoch不编造AP。20分钟自动更新。

## 最新绘图频率：每分钟

用户要求以后每1分钟更新。所有 plot_* 脚本的 --watch 刷新间隔改为60秒，现有后台绘图进程已重启；训练数据、训练配置、优化器状态不因此改变。epoch指标依然只在完整epoch结束后产生，轮内loss可随日志更新。此前20分钟绘图规则由此替代。

## 2026-09-30：浅层细节分支及时止损与轻量方法审计

用户要求及时止损。detail800 在17个完整epoch后停止，最佳训练epoch13的AP=53.8207，最新开训前实测AP=54.1861；停止控制器、训练及绘图watch进程，已确认无该配置的训练进程残留。STOPPED.json保留原因及轮次，已有模型与日志保留；未标记COMPLETE。

已完成：TXT转换往返AP无损检查；baseline整图640/800/960/1120；按类别NMS(0.5/0.6/0.7)、线性Soft-NMS(0.5/0.7)；整图+四局部视图切片(0.5/0.6的部分组合)；640/800持续增强低LR微调各20轮。800微调进一步复测640/800/960/1120与640/800切片，同样包含上述6种后处理。原权重与微调权重均显示整图原始输出更优；切片可提高小目标AP但损害整体AP。

收益：原baseline验证最佳53.9787，800微调54.1954(约+0.22)；局部裁剪15轮最佳54.4786，但未做用户明确跳过的普通续训对照，不作裁剪独立收益归因，也未自动迁移全量。

尚未完成/不能声称穷举：EMA与同轮原始权重对照、独立LR/weight-decay/增强强度扫描、单模型翻转TTA、裁剪比例扫描、困难样本重采样；已有EMA/warmup/cosine/裁剪等训练技巧不等于已做各项消融。phase2正式得分仍未知，全量训练不在已见val400上作独立选优。下一次改动优先以具体错误诊断和正式提交反馈为依据。

## 用户要求：翻转TTA、LR/增强小范围调整、Soft-NMS

- 单模型水平翻转TTA：同一权重推理原图和翻转图，先恢复X坐标，再类别内合并；无多模型或多阶段权重融合。缓存加入翻转标识，普通推理缓存保持兼容。
- 推理对照使用 ft_aug800 第20轮权重：原图与原图+水平翻转；线性Soft-NMS IoU阈值0.3/0.5/0.6/0.7，Gaussian Soft-NMS sigma0.3/0.5/0.7，同时保留原始输出和hard NMS0.5/0.6/0.7。最多100框；不以TTA原始重复框输出为默认提交策略。
- 四组8epoch训练为2x2网格，均从同一ft_aug800第20轮权重开始：LR5e-6/1.5e-5，backbone为1%；原增强(photometric .5、zoom .5、IoU crop .8)与温和增强(.2/.2/.3)。水平翻转与多尺度范围保持同配置，有效batch9，EMA、1轮warmup、cosine、8.5GiB限额。
- GPU0/2/5跑原增强低LR然后高LR；GPU1/3/6跑温和增强低LR然后高LR；GPU7做推理与各训练候选最佳权重的同模型TTA对照。GPU4空闲显存低于门槛，未使用。
- 两组低LR三卡冒烟检查已通过；GPU7推理对照与训练并行。CPU测试通过翻转坐标双重恢复、类别内Gaussian衰减、不修改原始预测、TXT往返。
- plots/light_tuning对应monitoring/light_tuning：每1分钟更新总体/尺寸/类别AP、loss、LR和已完成的推理对照。完整比较保存在experiments/light_tuning/comparison.json；不把本地增益当成phase2分数。

## 用户恢复要求：补跑裁剪的普通续训对照

- 重新授权continue800_control，15轮、LR1e-5、三卡有效batch9，同一ft_aug800第20轮权重、seed、EMA、warmup/cosine、原增强与800多尺度；与targetcrop800配置相比仅crop概率从.35改为0和输出目录不同，配置一致性断言通过。
- 曾取消的control目录无metrics.jsonl/COMPLETE，只保留了初始化/冒烟输出，已移到runs/continue800_control_cancelled_时间戳，不删除或混入新结果。
- queue_crop_control.py等待GPU0/2/5上的tune800_high_standard完成并释放显存，随后自动调用原targetcrop控制器；已完成的裁剪不重跑。避免与仍在跑的四组调参抢卡，保持相同三卡条件。
- 完成后生成裁剪/续训完整对照及最后5轮总体、小目标、常见类别的相对筛选结果；不自动重新启动已止损的细节分支。
- plot_crop_control.py每分钟更新monitoring/crop_control总体、类别、loss曲线；此前跳过对照的决定由当前用户请求替代。

## 首次phase2反馈：TTA + Gaussian Soft-NMS 得分49.986

用户明确确认49.986对应ft2000_aug800 epoch20、800水平翻转TTA、Gaussian Soft-NMS sigma .7的包。该得分为用户报告的正式结果，尚无同模型800原图phase2分数，不能把低分单独归因于Soft-NMS或TTA。val400上的54.95及+.79未直接迁移到phase2。

紧急替代：同权重800原图包已完整可提交；另生成960整图无TTA/无Soft-NMS包作为更难小目标场景候选。对应val400整体53.6949（相比800原图-.4706）、小目标29.8811（+2.4052）。该取舍有记录，不保证phase2更高；不依据已见val400评选全量模型独立最优。现有LR/增强实验继续，不把小幅验证波动称为突破。

## 第二次phase2反馈：960原图得分49.813

用户通过对应提交包标注确认960原图包得分49.813，比800翻转+Gaussian Soft-NMS包49.986低0.173。两套设置均未达到55目标。由于两次同时改变了分辨率与后处理，而800原图/原始2000 baseline640的正式成绩尚未报告，不能据此单独证明TTA、Soft-NMS或800微调的负面影响。

下一轮提交优先补原始2000 baseline640与当前2000微调800原图的模型对照，区分训练迁移与推理设置，避免继续仅按val400排序消耗提交。LR低档原增强最佳54.3567但最终54.1682，温和增强最终54.0958，尚无明显稳定收益，不急于推广全量；高档两组与裁剪续训对照依当前队列继续。


## 2026-10-01：目标57、困难场景诊断和真正的P2检测层

用户报告两个phase2分数49.986/49.813，当前已无提交机会，明确要求继续改善至至少57。四组LR/增强试验均完成8轮：最佳训练AP54.3567/54.0958/54.2781/54.3001，尚无足够稳定收益支持全量推广。普通800续训对照正在GPU0/2/5补跑；不重启已止损的detail800。

现有ft_aug800 epoch20、800原图预测的val400诊断：整体54.1655；GT框数≥10的密集子集109张/1934框，AP50.2349；其余291张/1096框，AP58.9147；小目标占比≥50%且至少3框的子集23张/313框，AP39.6085。子集重叠、类别组成不同、困难小目标子集很小；结果只是定位线索，不证明phase2低分原因。逐类别IoU.5/score.05贪心覆盖审计与COCO AP分别报告，不混作官方召回。

新候选P2HybridEncoder把stride4特征作为独立检测层交给四层decoder，与旧detail800的浅层融合不同。保留旧三层encoder参数；上采样P3加零初始化门控的浅层支路产生P2，新增支路使用GroupNorm，旧encoder BN冻结运行统计。迁移decoder三层输入投影、12→15个采样点及注意力参数，新增P2注意力bias=-2。anchor宽度随实际stride缩放，旧8/16/32层宽度保持不变。包含BN处理、尺度等变化，因此不称为纯结构消融。

P2从ft_aug800 epoch20权重迁移，训练640、多尺度峰值800、20轮、2轮warmup/cosine、旧head LR2e-5/backbone2e-7，新支路1e-4、门控1e-3，有效batch9（每卡1、累积3），EMA和常规增强；GPU1/3/6，8.5GiB进程显存上限。零门控、严格权重迁移、旧注意力参数和anchor保持、CPU有限输出/优化器参数覆盖、真实数据加载检查通过；三卡峰值尺度12batch冒烟通过，包含rank2单批空GT。正式训练已启动，初始640 AP52.4858，与原模型同尺度52.4248接近；不能与800 AP直接作结构归因。

scripts/after_p2.py独立等待训练COMPLETE，完成后在GPU7按空闲显存门槛评估P2最佳训练epoch和原模型，各自640/800原图推理，并跑相同密集/小目标子集诊断。输出最后5轮稳定性、同尺度总体/小目标/子集/常见类别变化；不自动按局部验证最优迁移全量或声明达到57。评测失败保留状态，不自动重启训练。scripts/plot_p2.py每60秒更新图表；AP仅来自完整epoch。


## 2026-10-01：用户授权夜间并行DEIMv2与上下文局部训练

用户确认RF-DETR-L已试过，弱于D-FINE-X，移出候选；授权试DEIMv2-L/X，并同时开展整图与保留上下文的局部训练。用户将睡觉、不再回复，要求成功启动并利用夜间时间；不依赖追加确认。P2按最新指示继续，不因新实验终止。

已完成裁剪对照：targetcrop800最后5轮AP中位54.3196，普通续训53.8400（+0.4796）；小目标+0.7169；常见类别无>2点下降。此前相对普通续训的裁剪筛选通过，但不是phase2提升或突破证据。

新增ContextTargetViews：每张训练图仍等概率进入训练；约65%保留整图，35%在存在小目标时取宽高比例.55–.8的局部视图，完整保留所选锚目标。局部锚选择按小目标尺寸与邻近目标密度加权，各权重有上限；不使用测试数据、预测伪标签或外部训练图。其他可见目标框正常裁剪并保留标签，空/尺寸不适合时退回整图。光度增强/翻转/800多尺度仍使用；不再对整图分支做ZoomOut/IoUCrop。因此新contextfull800是严格配对整图控制，不能直接与旧普通增强控制作纯采样归因。两组同起点ft_aug800 epoch20、seed、LR1e-5/backbone1e-7、EMA.999、有效batch9、sync_bn=False/find_unused=True、12轮；先contextmix800后contextfull800，GPU0/2/5。边界框/字段/面积/输入不变检查通过，三卡峰值992、6batch（含rank2单批空GT）冒烟通过。

DEIMv2官方源revision 1d2ca42171570c713e78fc6a766ec5104b7f4724保存在忽略目录experiments/model_sources，prepare_deimv2.py可重建并核验；L/X官方COCO safetensors来源Intellindust Hugging Face，SHA256记录在脚本及本地provenance。完整DINOv3+STA+encoder+decoder加载，AICOMP12类分类/去噪embedding重置，恢复safetensors去重的up/reg_scale共享别名，严格拒绝其余丢失/尺寸不匹配。模型构造器的“from scratch”日志先于完整检测权重加载，不代表随机骨干开训。训练/推理均本地完成。

DEIMv2-L在GPU7单卡batch6×累积2；X在GPU4单卡batch4×累积3；两者有效batch12、640多尺度480–800、24轮、2轮warmup/cosine、head LR1.5e-4/backbone5e-6、EMA.999、AMP/TF32、clip.1、8GiB上限，最后4轮关闭强增强与多尺度。保留官方ImageNet归一化和DEIMCriterion，不使用Mosaic/Mixup/CopyBlend作为首轮模型比较的额外变量；非严格同条件架构归因。通过峰值800的实际训练/EMA/验证测试，L峰值6650.8MiB、X6727.3MiB；正式训练已启动。第一轮12类分类从零适配，不能要求立即达到原模型已训练100+20轮的AP。

run_night_experiments.py分别管理三条队列和失败状态，已有部分训练时拒绝覆盖；上下文两组完成后自动生成最后5轮总体/小目标/常见类别的配对筛选。plot_night_experiments.py每60秒更新总体mAP@50-95、尺寸/类别AP、loss/LR及进度，完整epoch之外不编造AP。输出与权重/数据继续Git忽略。DEIM运行GPU7期间P2后续离线评估会按空闲显存门槛等待，不叠加到训练卡。


夜间启动补充：L已完成首轮训练、完整val400评估（12类指标）及模型/EMA/优化器/调度状态保存，并进入第2轮；上下文实验也完成首轮及权重保存。新分类头首轮AP低，不作为开训失败或新模型已超过baseline的证据。evaluate_variants.py新增独立deimv2 backend，使用官方ImageNet归一化，缓存身份包含backend，严格加载EMA权重并重建尺度buffer；800尺度两张验证图的完整预测/TXT往返检查通过。after_night_models.py等待L/X各24轮完成，自动比较最佳轮640/800及最终轮640，并生成相同val400密集/小目标子集诊断；某候选训练失败时记录并继续检查另一候选，不对失败模型生成结果。


## 2026-10-01：论文与底层机制审计，纠正负面实验归因

用户要求查阅论文、分析现有问题并深入考虑突破路线。本轮做离线诊断，不修改已完成训练、不训练测试、不生成使用真值修正的提交。

1. 训练状态漂移已得到配对推理证据：contextfull800最后一轮非EMA权重800原图AP52.7106；保持全部EMA模型参数不变，仅替换276个非backbone BN buffer为ft_aug800 epoch20的值，AP54.1927、小目标26.9608。训练阶段从SyncBN有效batch9变成每卡BN batch3，以及增强分布改变，均可能导致统计漂移；该干预证明BN状态对退化有实质贡献，未证明只改某一个训练因素的独立因果。全量替换统计量不是新模型收益，不自动提交；下一轮应做冻结BN运行统计的配对训练，保留affine可训练，不能把梯度累积视为BN大batch。参考Group Normalization ECCV2018 https://arxiv.org/abs/1803.08494 。

2. TIDE对固定ft_aug800 epoch20、800原图、val400 top100预测在IoU.50–.95分别作理想错误干预。TIDE平均AP54.1656与原COCO54.1655一致。单独理想修复各错误的平均dAP：Loc32.5882、Bkg2.7759、Miss.9333、Cls.5708、Both.2643、Dupe.0291。数值不是可训练获得的收益，不能加和；用于判断定位而非重复框/错分类是已知验证集主要损失。AP50=81.65、AP75=55.09、AP90=21.48，后处理框精修失败不能否定端到端定位监督。参考TIDE ECCV2020 https://arxiv.org/abs/2008.08115 ，源码 https://github.com/dbolya/tide 。

3. 新验证相似性检查：400张val对1600张train，63bit pHash距离≤4且32x32 RGB RMSE≤.05的候选38张；查看最相似12对，确认多对几乎同场景/邻帧。候选38图AP55.9176，余362图AP54.3804（类别分布不同且部分缺类）；排除这些候选未降低AP，因此不能将本地/榜单差距归因于这38图。后续按场景/序列划分仅作为额外泛化对照；当前seed20260929主划分不覆盖。源数据不变。检查原始未裁剪GT语义：val400中83框发生裁剪，按原坐标评估AP54.1227，比54.1655低.0428，也不足以解释phase2差距；官方GT处理尚未知。

4. DEIMv2首轮24epoch迁移删掉了官方Dense O2O的Mosaic/Mixup/CopyBlend，重置12类分类头，并非完整论文训练机制复现；D-FINE已接受100+20轮且预训练含Objects365→COCO，DEIMv2起点COCO且训练预算不同，不能宣称公平架构优劣。DEIM论文强调Dense O2O增加每图正样本和MAL处理低质量匹配。下一条监督路线优先保留已成熟D-FINE权重，分别检验MAL/密集监督及匹配改进，控制BN与增强；不再次盲目换整套骨干。参考 https://arxiv.org/abs/2412.04234 ，DEIMv2 https://arxiv.org/abs/2509.20787 。

5. 当前D-FINE已有FDR、GO-LSD、IoU目标VFL以及基于角点分布的LQE，不能把增加“质量头/分布回归/自蒸馏”当成新机制。代码中VFL正样本权重为IoU、FGL权重亦与IoU相关；小目标低质量匹配可能获得较弱监督，这是待梯度验证的机制假设。优先试保留IoU/GIoU与FGL主损失、加入有上限的尺度归一化中心/边长辅助误差，并仅将NWD作为小目标匹配辅助成本；避免替换最终IoU评价或压倒高IoU优化。参考NWD https://arxiv.org/abs/2110.13389 ，Align-DETR https://arxiv.org/abs/2304.07527 ，D-FINE https://arxiv.org/abs/2410.13842 。

优先顺序：冻结BN控制恢复可靠实验条件 → 小目标/细长框的端到端高IoU定位监督及匹配 → 保留D-FINE权重的Dense O2O/MAL机制对照；每轮报告AP75/AP90、小目标、密集子集与类别收益，不能只看单轮总体峰值。验证Oracle只做定位线索，不进入预测包。IR/Depth若重试，必须先证明对齐和有效互补，不能重复直接拼接或事后框修正。


审计反例与几何补充：将DEIMv2-X的279个BN buffer恢复为官方COCO预训练统计量，800推理AP由48.2976降至46.9929、小目标22.1672；因此BN移植只解释contextfull退化，不能推广为所有模型的修复。800输入下val400有212框短边<8像素、825框短边<16像素（共3030）；短边百分位P10/P25/P50/P75为9.58/15/27.08/48.75像素。COCO原图面积small定义与网络实际可分辨短边不同，应同时看细长框、相对中心/边界误差。两个诊断均保存在experiments/mechanism_audit，任何val真值理想修正预测只作误差分析。

落实下一轮控制设计：同一ft_aug800 epoch20权重、相同640/800尺寸和增强、有效batch9、每卡3/三卡，先比较冻结BN运行统计与原设置；确认稳定后进行2x2的匹配/定位监督对照（原IoU/L1；小目标NWD辅助匹配；有上限尺度归一化辅助定位；二者组合），保留原GIoU/FGL/GO-LSD/LQE。训练主指标增加AP75及逐IoU AP/短边子集，检查高IoU提升和常见类损伤。若小目标匹配本来良好，跳过NWD；若主要错误集中在GT歧义，先看训练标注质量，不能改测试。Dense O2O/MAL作为另一独立机制对照，不能把所有新损失一次叠入且失去归因。训练统计冻结和新损失均尚未开训，当前只是有证据的候选设计。

匹配机制实测补充：scripts/audit_matching.py在32张带原有增强的train1600视图上运行训练模式decoder，冻结BN且不更新梯度/权重。最终层182个正查询、跨层GO联合206个查询；最终层正查询与GO联合指向不同GT或类别的数量均为0。最终层匹配平均IoU .81565，其中短边<16像素的54个匹配平均IoU .68829；12个匹配IoU<.5、43个<.75。样本量有限，不能证明全训练无跨层冲突，但不支持把匹配冲突当作当前首要问题。因此上述2x2匹配网格暂缓，优先冻结BN的配对控制与尺度归一化定位辅助损失；NWD只有在扩大诊断后发现匹配不稳时再纳入。诊断保持原匹配器与损失，不作验证/测试标签训练。

复现入口：AICOMP环境运行scripts/audit_training_mechanisms.py生成BN候选/训练验证相似性报告，scripts/tide_diagnostics.py生成逐IoU错误分解，scripts/audit_matching.py --batches 16复现本次匹配抽查。TIDE依赖在requirements_diagnostics.txt中单独固定，不升级现有训练的NumPy/OpenCV。诊断输出、权重、数据和近邻图继续Git忽略；所有数值为本地诊断，phase2已知正式结果仍为49.986/49.813。

## 2026-10-01：用户授权子代理并行启动机制实验

用户明确要求派发subagent并行开始上述方向，成功启动后代理可结束。三个子代理负责BN配对控制、定位辅助损失、MAL/密集增强；root统一接入逐IoU指标、绘图、完成后比较及Git管理。五项均从ft_aug800 epoch20的成熟EMA权重开始，official train1600训练/val400验证，不使用test训练，不直接启动全量推广。

统一配置：contextfull800整图光度增强/翻转、800多尺度峰值992、12轮、head LR1e-5/backbone1e-7、warmup1/cosine至.1、EMA.999/warmups1000、单卡每批3张×累积3有效batch9、val batch2、seed20260929、workers2、sync_bn=false、PyTorch 8.5GiB显存上限。GPU0 bn_adapt800、GPU1 bn_frozen800、GPU5 relative_box800、GPU6 mal800、GPU4 dense_o2o800。启动前按可用显存门槛等待，不终止其他用户任务；驱动统计的本任务显存约9GiB（含框架外CUDA开销）。实际共享计算资源可能降低速度，不承诺不影响其他任务的吞吐。

- BN配对仅输出目录与freeze_bn_statistics不同；冻结BN运行统计、affine仍可训练，model.train反复调用及EMA均兼容。数值检查通过。实际两组峰值992 smoke通过，单任务峰约7746MiB。首轮实际权重检查：冻结对照及定位实验的276个非骨干BN buffer与起点逐项完全一致，last包含优化器/scaler/EMA/调度状态。
- relative_box800只增加尺度归一化辅助定位项，原matcher、L1/GIoU/VFL/FGL/GO-LSD/LQE保留。公式 .25/N_GO × sum mean4 SmoothL1((pred−GT)/[max(w,.02),max(h,.02),max(w,.02),max(h,.02)], beta=.1；归一化cxcywh，分母下限防微小框梯度失控，SmoothL1尾部线性，误差本身没有截断。每个原回归分支应用自己的.25权重，不重复乘bbox×5；final/aux/pre/encoder/DN、空GT、零误差、同位移小框惩罚更大、有限梯度检查通过。
- mal800只将VFL分类公式替换为官方MAL，保留原分类分支调度/权重与全部定位损失。正类soft target=IoU^gamma、正项权重1，负类target0、权重sigmoid(logit)^gamma；采用官方base/deim.yml gamma1.5、mal_alpha=null，matcher gamma仍2。正常/空GT/values分支与本地官方函数逐值完全一致，梯度有限。来源DEIMv2官方revision 1d2ca42171570c713e78fc6a766ec5104b7f4724的engine/deim/deim_criterion.py；不是完整DEIM复现。
- dense_o2o800是像素保留局部四图密集增强的单因素变体：概率.25全程不变，四个训练源各映射800后取目标中心400块拼成800，75%整图；相对baseline800保留未截断目标像素尺寸，裁剪至少保留原框面积.3。其他源仅来自同一train1600的raw load_item，绝不读取val/test。64图抽样原均4.17框、拼图均16.36框；合成20px框尺寸、边界/labels/area/iscrowd/空GT检查通过。不是四整图压缩Mosaic，也不是完整Dense O2O/Mixup/CopyBlend配方。

五项真实峰值6batch训练+验证检查均通过，全部已完成开训前400图评测及至少3次正式优化器更新，loss有限。不同GPU/TF32计算起点AP约54.16–54.20，不能把微小起点差异当机制收益。首轮冻结对照AP54.2819/AP90 21.6114、定位AP54.1379/AP90 21.7664；仅一轮结果不作收益结论，继续12轮。所有子代理在启动检查后结束，后台控制器/训练进程独立持久运行。

存储保护：根分区初始仅约19GiB可用，统一保留逐轮完整metrics、best/last完整训练状态，compact推理权重仅第3/6/9/12轮；不重复写full epoch状态。跨任务flock串行checkpoint落盘，写前要求足够本次tensor字节+512MiB余量，防多任务临时文件同时挤满磁盘；既有数据/权重未删除。

启动事件纠正：MAL第一次配方重启时只停止控制器，残留独立训练进程，导致两个同run worker短暂并存。root发现后仅停止自身MAL控制器和两个训练进程组，整个污染run/log/status归档experiments/mechanisms/archive/mal_duplicate_20261001_115717，不进入比较；其他四项未中断。干净MAL从原父权重gamma1.5重启，确认GPU6唯一CUDA训练PID及3次更新。共同训练入口新增worker自身flock；Dense/MAL启动器新增同配置活跃进程拒绝、TERM/INT时只清理自身child组、延迟signal handler避免spawn竞态和子进程继承屏蔽信号。隔离的真实父子进程退出回归检查通过；公共worker duplicate拒绝也在CUDA初始化前检查通过。

scripts/extra_iou_metrics.py利用同一COCO evaluator的precision追加IoU.50–.95逐阈值AP、逐类AP90，不增加推理。scripts/plot_mechanism_experiments.py每60秒刷新monitoring/mechanism_trials中的总体/类别/损失/高IoU曲线及轮内进展，只完整epoch产生验证曲线。scripts/after_mechanism_experiments.py等待完成/明确失败后比较最后5轮总体、小目标、AP75/AP90、类别变化；GPU7按显存门槛依次复测保存best的800原图、密集/小目标子集，增加800下短边<16占比≥.5且至少3目标的40图/606框子集，以及排除近邻候选后的362图验证。辅助子集复现检查：共同父权重AP40.9004/54.3804，与既有相似性报告一致；子集重叠、类别不同，不推算phase2表现、不自动推广全量。
