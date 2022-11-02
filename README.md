# NLP第二次作业

这是国科大研究生课程《自然语言处理》的第二次作业，用三种神经网络构建语言模型，这里使用了NNLM，RNN和Attention，使用Pytorch实现。

code文件夹中包含了代码文件，其中config文件夹中为配置文件，dataset文件夹中包含了数据文件和数据处理脚本，model文件夹中包含了模型定义文件，result文件夹包含了保存的模型参数和实验的日志文件。main.py是训练和测试文件，utils.py是辅助函数文件。

## 运行
要想运行，首先将news.2018.zh.shuffled.deduped.gz文件保存到dataset文件夹中，然后运行

```bash
$> python main.py --config $CONFIG_FILE
```
其中 $CONFIG_FILE 可取的值包括 "./config/NNLM.yml", "./config/RNNtext.yml", "./config/Atten.yml"

实验的log和模型参数将被保存到result文件夹中的对应位置。