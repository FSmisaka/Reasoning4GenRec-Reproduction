# SIDReasoner 思维链多样性报告

数据集: **Video_Games** | 生成时间: 2026-10-30 09:54:41 | 统计参数: top=25, pair_cap=600, max_docs=2000, seed=42, equalize=on, device=cpu

- text 视角 = 剔除 SID token 后的自然语言; raw 视角 = 保留 SID token。
- 关键读法: 成对余弦均值/完全重复率/首句 Top 覆盖越高、distinct 与 gzip 率越低, 说明思维链越同质化。
- equalize=on 时各语料已采样到与最小语料相同的篇数(语料级指标才可跨语料对比); 每个语料的采集总量见其小节。
- 词汇类指标/gzip/Jaccard 均有 含停用词(默认) 与 去停用词(仅内容词, *_content / '去停用词' 对照) 两套口径。

### 汇总对比 · text 视角

| 语料 | n | 词数均值 | TTR | dist-2(语料) | 完全重复率 | 余弦均值 | 余弦p95 | 余弦>0.8 | Jaccard均值 | 首句Top1 | 首句Top5 | gzip率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage2 | 300 | 147.2 | 0.0240 | 0.1390 | 0.7% | 0.2846 | 0.4314 | 0.0% | 0.0639 | 1.3% | 6.7% | 0.167 |
| stage3 | 300 | 151.1 | 0.0242 | 0.1429 | 0.7% | 0.2786 | 0.4231 | 0.0% | 0.0625 | 5.3% | 12.3% | 0.169 |
| teacher | 300 | 154.5 | 0.0449 | 0.2958 | 0.0% | 0.1894 | 0.2697 | 0.0% | 0.0162 | 0.3% | 1.7% | 0.232 |

### 汇总对比 · raw 视角

| 语料 | n | 词数均值 | TTR | dist-2(语料) | 完全重复率 | 余弦均值 | 余弦p95 | 余弦>0.8 | Jaccard均值 | 首句Top1 | 首句Top5 | gzip率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage2 | 300 | 159.9 | 0.0334 | 0.1803 | 0.7% | 0.2097 | 0.3127 | 0.0% | 0.0561 | 1.3% | 6.7% | 0.167 |
| stage3 | 300 | 162.8 | 0.0334 | 0.1813 | 0.7% | 0.2098 | 0.3139 | 0.0% | 0.0558 | 5.3% | 12.3% | 0.169 |
| teacher | 300 | 164.4 | 0.0527 | 0.3221 | 0.0% | 0.1588 | 0.2272 | 0.0% | 0.0149 | 0.3% | 1.7% | 0.232 |

### 汇总对比 · text · 去停用词 视角

| 语料 | n | 词数均值 | TTR | dist-2(语料) | 完全重复率 | 余弦均值 | 余弦p95 | 余弦>0.8 | Jaccard均值 | 首句Top1 | 首句Top5 | gzip率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage2 | 300 | 147.2 | 0.0365 | 0.2500 | 0.7% | 0.2846 | 0.4314 | 0.0% | 0.0270 | 1.3% | 6.7% | 0.158 |
| stage3 | 300 | 151.1 | 0.0371 | 0.2562 | 0.7% | 0.2786 | 0.4231 | 0.0% | 0.0263 | 5.3% | 12.3% | 0.162 |
| teacher | 300 | 154.5 | 0.0669 | 0.4893 | 0.0% | 0.1894 | 0.2697 | 0.0% | 0.0053 | 0.3% | 1.7% | 0.223 |

## 语料: stage2
- 来源: generate:runs/sidreasoner/Video_Games_stage2_activation/final_checkpoint | 采集 300 篇, 统计 300 篇 | 思维链完整闭合率 100.0%
- 文档数 300 | 词数均值 147.2 (p5 125.0, p95 170.1) | 词表 1059 | TTR 0.0240 | gzip率 0.167
- 语料级 distinct-1/2 = 0.0240 / 0.1390; 文档级均值 0.6395 / 0.9389; unigram 熵 7.4 bits (归一化 0.739), bigram 熵 10.4 bits (归一化 0.826)
- 完全重复率 0.7% (298/300 篇唯一);
  - 成对余弦(text): 均值 0.2846, 中位 0.2810, p95 0.4314, >0.6 0.1%, >0.8 0.0% | 3-gram Jaccard: 均值 0.0639, >0.5 0.0%
  - 成对余弦(raw): 均值 0.2097, 中位 0.2056, p95 0.3127, >0.6 0.0%, >0.8 0.0% | 3-gram Jaccard: 均值 0.0561, >0.5 0.0%
- 去停用词对照(text, 仅内容词): 词表 999 | TTR 0.0365 | distinct-1/2(语料) 0.0365 / 0.2500 | unigram 熵 7.8 bits (归一化 0.779), bigram 熵 11.0 bits (归一化 0.861) | 3-gram Jaccard 均值 0.0270 | gzip率 0.158
- 首句模板: 唯一率 90.0%, Top1 覆盖 1.3%, Top5 覆盖 6.7%
  - (4x, 1.3%) i observe that the user has a strong inclination towards action-adventure and first-person shooter genres as evidenced b
  - (4x, 1.3%) i observe that the user has a strong inclination towards action-adventure and open-world games as evidenced by their eng
  - (4x, 1.3%) i observe that the user has a strong inclination towards action-packed and competitive gaming experiences as evidenced b
  - (4x, 1.3%) i notice a strong inclination towards action-adventure and open-world games in the user's history particularly those tha
  - (4x, 1.3%) i notice a strong inclination towards action-adventure and rpg genres in the user's history particularly with titles lik

**TF-IDF 平均权重 Top15 (text 视角)**

| 词 | 平均TF-IDF权重 | 出现文档数 |
|---|---|---|
| user | 0.0787 | 300 |
| games | 0.0759 | 264 |
| action | 0.0724 | 233 |
| gaming | 0.0715 | 214 |
| titles | 0.0701 | 262 |
| gameplay | 0.0663 | 290 |
| towards | 0.0559 | 279 |
| may | 0.0553 | 300 |
| experiences | 0.0545 | 264 |
| offer | 0.0539 | 293 |
| enjoy | 0.0538 | 293 |
| character | 0.0514 | 186 |
| narratives | 0.0489 | 221 |
| immersive | 0.0487 | 264 |
| elements | 0.0471 | 192 |

**原始词频 Top15 (text 视角, 去停用词)**

| 词 | 次数 | 文档数 |
|---|---|---|
| user | 855 | 300 |
| gaming | 728 | 214 |
| games | 713 | 264 |
| gameplay | 629 | 290 |
| titles | 591 | 262 |
| action | 571 | 233 |
| may | 495 | 300 |
| towards | 457 | 279 |
| enjoy | 456 | 293 |
| offer | 454 | 293 |
| experiences | 423 | 264 |
| immersive | 351 | 264 |
| strong | 348 | 289 |
| preference | 316 | 296 |
| particularly | 311 | 249 |

**Bigram 频率 Top15 (text 视角)**

the user(767), games that(519), that offer(424), may enjoy(374), titles that(358), that the(339), preference for(314), a strong(302), a preference(295), user has(288), their gaming(260), inclination towards(246), additionally the(242), the user's(241), strong inclination(237)

**Bigram 频率 Top15 (raw 视角, 含 SID)**

`the user`(767), `games that`(519), `that offer`(424), `may enjoy`(374), `titles that`(358), `that the`(339), `preference for`(314), `a strong`(302), `a preference`(295), `user has`(288), `their gaming`(260), `inclination towards`(246), `additionally the`(242), `the user's`(241), `strong inclination`(237)

**最相似样本对(余弦 Top, 截断 200 字符)**

1. 余弦 1.0000
   - A: I observe that the user has a strong inclination towards gaming accessories, particularly those that enhance the experience of the Wii console. Their repeated interactions with various models of the W
   - B: I observe that the user has a strong inclination towards gaming accessories, particularly those that enhance the experience of the Wii console. Their repeated interactions with various models of the W

2. 余弦 1.0000
   - A: I observe that the user has a strong inclination towards gaming accessories, particularly those that enhance the experience of the Wii console. Their repeated interactions with various models of the W
   - B: I observe that the user has a strong inclination towards gaming accessories, particularly those that enhance the experience of the Wii console. Their repeated interactions with various models of the W

3. 余弦 1.0000
   - A: I observe that the user has a strong inclination towards gaming accessories, particularly those that enhance the experience of the Wii console. Their repeated interactions with various models of the W
   - B: I observe that the user has a strong inclination towards gaming accessories, particularly those that enhance the experience of the Wii console. Their repeated interactions with various models of the W

## 语料: stage3
- 来源: generate:runs/sidreasoner/Video_Games_stage3/actor_merged_1900 | 采集 300 篇, 统计 300 篇 | 思维链完整闭合率 100.0%
- 文档数 300 | 词数均值 151.1 (p5 132.0, p95 180.1) | 词表 1096 | TTR 0.0242 | gzip率 0.169
- 语料级 distinct-1/2 = 0.0242 / 0.1429; 文档级均值 0.6321 / 0.9350; unigram 熵 7.5 bits (归一化 0.739), bigram 熵 10.5 bits (归一化 0.826)
- 完全重复率 0.7% (298/300 篇唯一);
  - 成对余弦(text): 均值 0.2786, 中位 0.2763, p95 0.4231, >0.6 0.1%, >0.8 0.0% | 3-gram Jaccard: 均值 0.0625, >0.5 0.0%
  - 成对余弦(raw): 均值 0.2098, 中位 0.2071, p95 0.3139, >0.6 0.0%, >0.8 0.0% | 3-gram Jaccard: 均值 0.0558, >0.5 0.0%
- 去停用词对照(text, 仅内容词): 词表 1037 | TTR 0.0371 | distinct-1/2(语料) 0.0371 / 0.2562 | unigram 熵 7.8 bits (归一化 0.780), bigram 熵 11.1 bits (归一化 0.864) | 3-gram Jaccard 均值 0.0263 | gzip率 0.162
- 首句模板: 唯一率 77.3%, Top1 覆盖 5.3%, Top5 覆盖 12.3%
  - (16x, 5.3%) i notice a strong inclination towards action-packed and immersive gaming experiences in the user's history
  - (8x, 2.7%) i notice a strong inclination towards action-adventure and rpg genres in this user's history particularly with titles li
  - (5x, 1.7%) i notice a strong inclination towards action-adventure and rpg genres in the user's history as evidenced by their engage
  - (4x, 1.3%) i notice a strong inclination towards action-adventure and immersive storytelling in the user's history
  - (4x, 1.3%) i notice a strong inclination towards action-packed games with immersive narratives and engaging gameplay mechanics in t

**TF-IDF 平均权重 Top15 (text 视角)**

| 词 | 平均TF-IDF权重 | 出现文档数 |
|---|---|---|
| user | 0.0770 | 300 |
| games | 0.0754 | 259 |
| action | 0.0708 | 224 |
| gaming | 0.0688 | 224 |
| titles | 0.0687 | 250 |
| gameplay | 0.0658 | 283 |
| experiences | 0.0557 | 265 |
| may | 0.0549 | 299 |
| offer | 0.0547 | 291 |
| enjoy | 0.0505 | 292 |
| character | 0.0502 | 172 |
| towards | 0.0501 | 279 |
| adventure | 0.0496 | 168 |
| elements | 0.0489 | 204 |
| immersive | 0.0485 | 262 |

**原始词频 Top15 (text 视角, 去停用词)**

| 词 | 次数 | 文档数 |
|---|---|---|
| user | 874 | 300 |
| gaming | 757 | 224 |
| games | 720 | 259 |
| gameplay | 628 | 283 |
| titles | 573 | 250 |
| action | 555 | 224 |
| may | 512 | 299 |
| offer | 473 | 291 |
| experiences | 452 | 265 |
| enjoy | 428 | 292 |
| towards | 407 | 279 |
| immersive | 363 | 262 |
| strong | 350 | 283 |
| preference | 333 | 297 |
| user's | 327 | 241 |

**Bigram 频率 Top15 (text 视角)**

the user(807), games that(523), that offer(422), titles that(369), may enjoy(357), preference for(325), that the(309), a preference(304), i notice(296), a strong(290), the user's(280), user has(252), user may(252), inclination towards(248), their gaming(246)

**Bigram 频率 Top15 (raw 视角, 含 SID)**

`the user`(807), `games that`(523), `that offer`(422), `titles that`(369), `may enjoy`(357), `preference for`(325), `that the`(309), `a preference`(304), `i notice`(296), `a strong`(290), `the user's`(280), `user has`(252), `user may`(252), `inclination towards`(248), `their gaming`(246)

**最相似样本对(余弦 Top, 截断 200 字符)**

1. 余弦 1.0000
   - A: I notice a strong inclination towards gaming accessories and consoles, particularly those related to the PlayStation 3 and Wii. The user has engaged with multiple variants of the same item, indicating
   - B: I notice a strong inclination towards gaming accessories and consoles, particularly those related to the PlayStation 3 and Wii. The user has engaged with multiple variants of the same item, indicating

2. 余弦 1.0000
   - A: I notice a strong inclination towards gaming accessories and consoles, particularly those related to the PlayStation 3 and Wii. The user has engaged with multiple variants of the same item, indicating
   - B: I notice a strong inclination towards gaming accessories and consoles, particularly those related to the PlayStation 3 and Wii. The user has engaged with multiple variants of the same item, indicating

3. 余弦 1.0000
   - A: I notice a strong inclination towards gaming accessories and consoles, particularly those related to the PlayStation 3 and Wii. The user has engaged with multiple variants of the same item, indicating
   - B: I notice a strong inclination towards gaming accessories and consoles, particularly those related to the PlayStation 3 and Wii. The user has engaged with multiple variants of the same item, indicating

## 语料: teacher
- 来源: train | 采集 300 篇, 统计 300 篇 | 思维链完整闭合率 100.0%
- 文档数 300 | 词数均值 154.5 (p5 129.9, p95 185.0) | 词表 2080 | TTR 0.0449 | gzip率 0.232
- 语料级 distinct-1/2 = 0.0449 / 0.2958; 文档级均值 0.6881 / 0.9692; unigram 熵 8.2 bits (归一化 0.743), bigram 熵 12.0 bits (归一化 0.875)
- 完全重复率 0.0% (300/300 篇唯一);
  - 成对余弦(text): 均值 0.1894, 中位 0.1862, p95 0.2697, >0.6 0.0%, >0.8 0.0% | 3-gram Jaccard: 均值 0.0162, >0.5 0.0%
  - 成对余弦(raw): 均值 0.1588, 中位 0.1554, p95 0.2272, >0.6 0.0%, >0.8 0.0% | 3-gram Jaccard: 均值 0.0149, >0.5 0.0%
- 去停用词对照(text, 仅内容词): 词表 2004 | TTR 0.0669 | distinct-1/2(语料) 0.0669 / 0.4893 | unigram 熵 8.7 bits (归一化 0.794), bigram 熵 12.7 bits (归一化 0.922) | 3-gram Jaccard 均值 0.0053 | gzip率 0.223
- 首句模板: 唯一率 100.0%, Top1 覆盖 0.3%, Top5 覆盖 1.7%
  - (1x, 0.3%) analyzing the user's interaction history i note the engagement with <sid><sid><sid> which indicates a strong preference 
  - (1x, 0.3%) i notice a clear affinity towards action-adventure and immersive gameplay experiences in the user's history with titles 
  - (1x, 0.3%) analyzing the user's interaction history it appears they favor action-adventure and first-person shooter genres highligh
  - (1x, 0.3%) based on the user's interaction history i can see a notable preference for narrative-driven games and adventure titles p
  - (1x, 0.3%) analyzing the user's interaction history i notice a strong inclination towards action-packed experiences that involve en

**TF-IDF 平均权重 Top15 (text 视角)**

| 词 | 平均TF-IDF权重 | 出现文档数 |
|---|---|---|
| user | 0.0602 | 299 |
| gameplay | 0.0583 | 293 |
| gaming | 0.0525 | 209 |
| games | 0.0507 | 262 |
| action | 0.0505 | 228 |
| titles | 0.0505 | 245 |
| experiences | 0.0466 | 272 |
| towards | 0.0421 | 226 |
| may | 0.0404 | 298 |
| immersive | 0.0396 | 263 |
| narratives | 0.0392 | 209 |
| engaging | 0.0381 | 226 |
| elements | 0.0371 | 192 |
| enjoy | 0.0368 | 258 |
| mechanics | 0.0363 | 189 |

**原始词频 Top15 (text 视角, 去停用词)**

| 词 | 次数 | 文档数 |
|---|---|---|
| user | 790 | 299 |
| gameplay | 730 | 293 |
| gaming | 582 | 209 |
| games | 533 | 262 |
| titles | 482 | 245 |
| experiences | 472 | 272 |
| action | 467 | 228 |
| may | 422 | 298 |
| immersive | 357 | 263 |
| towards | 351 | 226 |
| enjoy | 322 | 258 |
| engaging | 300 | 226 |
| preference | 293 | 268 |
| narratives | 292 | 209 |
| strong | 271 | 236 |

**Bigram 频率 Top15 (text 视角)**

the user(628), games that(302), preference for(288), may enjoy(249), titles that(239), interest in(225), a strong(214), that offer(200), user may(199), that the(189), the user's(184), a preference(174), they may(173), interaction history(165), engagement with(164)

**Bigram 频率 Top15 (raw 视角, 含 SID)**

`the user`(628), `games that`(302), `preference for`(288), `may enjoy`(249), `titles that`(239), `interest in`(225), `a strong`(214), `that offer`(200), `user may`(199), `that the`(189), `the user's`(184), `a preference`(174), `they may`(173), `interaction history`(165), `engagement with`(164)

**最相似样本对(余弦 Top, 截断 200 字符)**

1. 余弦 0.4728
   - A: I observe that this user has a strong inclination towards gaming peripherals, particularly high-quality controllers and mechanical keyboards, evidenced by the variety of Xbox controllers like <a_61><b
   - B: I notice that the user has shown a strong affinity for high-performance gaming hardware, particularly mechanical keyboards and gaming mice, as reflected in their interactions with items like <a_202><b

2. 余弦 0.4486
   - A: I notice a strong preference for amiibo and collectibles, particularly from the Animal Crossing series, as evidenced by the repeated interactions with various characters from that franchise. The user 
   - B: I observe a clear affinity for amiibo collectibles, especially from the Animal Crossing series, given the user's interactions with multiple characters such as <a_162><b_172><c_85>, <a_162><b_172><c_0>

3. 余弦 0.4379
   - A: I observe a strong inclination towards engaging action-adventure and multiplayer gaming experiences in this user's history. The presence of titles like <a_201><b_213><c_242> and <a_24><b_151><c_86> in
   - B: I notice that this user has a strong engagement with RPGs and action-adventure titles, particularly those that feature engaging narratives and immersive gameplay mechanics, as seen in <a_1><b_150><c_1
