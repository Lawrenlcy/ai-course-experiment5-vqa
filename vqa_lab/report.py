import collections
import json
import shutil
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .common import read_json,write_json,setup_plot,save_figure
from .evaluation import evaluate
from .data import image_path


def build_report(out,data_root='data'):
    out=Path(out);plt=setup_plot();figdir=out/'figures'
    manifest=read_json(out/'eval_manifest.json');ann={a['question_id']:a for a in manifest}
    detail=[];summary=[]
    for file in sorted(out.glob('predictions_*.jsonl')):
        method=file.stem.removeprefix('predictions_')
        predictions=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
        ev=evaluate(manifest,predictions)
        for p in predictions:
            counts=collections.Counter(a['answer'] for a in ann[p['question_id']]['answers'])
            detail.append(dict(p,method=method,score=ev.evalQA[p['question_id']],
                references=json.dumps(dict(counts)),majority=counts.most_common(1)[0][0]))
        for group in ['overall','yes/no','number','other']:
            n=len(predictions) if group=='overall' else sum(p['answer_type']==group for p in predictions)
            score=ev.accuracy['overall'] if group=='overall' else ev.accuracy['perAnswerType'].get(group)
            summary.append({'method':method,'group':group,'n':n,'accuracy_percent':score})
    df=pd.DataFrame(detail);table=pd.DataFrame(summary)
    df.to_csv(out/'per_question_results.csv',index=False,encoding='utf-8-sig')
    table.to_csv(out/'accuracy_summary.csv',index=False)
    order=[m for m in ['greedy','beam3','blank_image'] if m in df.method.unique()]
    fig,ax=plt.subplots(figsize=(11,5));groups=['overall','yes/no','number','other'];x=np.arange(4)
    for i,method in enumerate(order):
        values=[table[(table.method==method)&(table.group==g)].accuracy_percent.iloc[0] for g in groups]
        bars=ax.bar(x+(i-(len(order)-1)/2)*.24,values,.24,label=method)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=groups,ylim=(0,110),ylabel='Official VQA accuracy (%)',title='BLIP validation diagnostic — potential training overlap')
    ax.legend();save_figure(fig,figdir,'01_accuracy')
    counts=pd.Series([a['answer_type'] for a in manifest]).value_counts()
    fig,ax=plt.subplots(figsize=(8,4));bars=ax.bar(counts.index,counts.values,color='#2a6fbb');ax.bar_label(bars,padding=3)
    ax.set(title=f'Subset: {len(manifest)} questions / {len(set(a["image_id"] for a in manifest))} images',ylabel='Questions');ax.margins(y=.2)
    save_figure(fig,figdir,'02_distribution')
    timing=df.groupby('method')[['generation_seconds_per_question','e2e_seconds_per_question']].mean().reindex(order)
    timing.to_csv(out/'timing_summary.csv')
    fig,ax=plt.subplots(figsize=(9,4));(timing*1000).rename(columns={'generation_seconds_per_question':'Generation','e2e_seconds_per_question':'Load + preprocess + generate + decode'}).plot.bar(ax=ax,rot=0)
    ax.set(ylabel='Amortized ms/question',title='Batch throughput cost after warmup (not request latency)');ax.legend(fontsize=8)
    save_figure(fig,figdir,'03_timing')
    wide=df.pivot(index=['question_id','image_id'],columns='method',values='score')
    stats={};rng=np.random.default_rng(42)
    for method in order:
        delta=wide[method]-wide.greedy
        clusters=delta.groupby(level='image_id').agg(['sum','count'])
        ix=rng.integers(0,len(clusters),(3000,len(clusters)))
        boot=clusters['sum'].to_numpy()[ix].sum(1)/clusters['count'].to_numpy()[ix].sum(1)
        stats[method]={'delta_vs_greedy_pp':float(delta.mean()),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),
            'improved':int((delta>0).sum()),'unchanged':int((delta==0).sum()),'worsened':int((delta<0).sum()),
            'bootstrap':'3000 image-cluster paired resamples; seed 42'}
    write_json(out/'paired_statistics.json',stats)
    changed=pd.DataFrame(stats).T[['improved','unchanged','worsened']].drop(index='greedy',errors='ignore')
    if len(changed):
        fig,ax=plt.subplots(figsize=(9,4));changed.plot.bar(ax=ax,rot=0);ax.set(title='Paired changes relative to greedy',ylabel='Questions')
        save_figure(fig,figdir,'04_paired_changes')
    qt=df.groupby(['method','question_type']).score.agg(['mean','count']).reset_index()
    qt.to_csv(out/'question_type_summary.csv',index=False)
    g=qt[(qt.method=='greedy')&(qt['count']>=10)].sort_values('mean')
    fig,ax=plt.subplots(figsize=(9,max(4,len(g)*.4)));ax.barh([f'{r.question_type} (n={r.count})' for r in g.itertuples()],g['mean'])
    ax.set(xlim=(0,100),xlabel='Official VQA accuracy (%)',title='Question types with at least 10 examples')
    save_figure(fig,figdir,'05_question_types')
    case_dir=out/'case_images';case_dir.mkdir(exist_ok=True)
    base=df[df.method=='greedy'].sort_values('question_id')
    case_ids=[]
    for kind in ['yes/no','number','other']:
        case_ids.extend(base[(base.answer_type==kind)&(base.score==0)].head(2).question_id.tolist())
    success=[]
    for kind in ['yes/no','number','other']:
        success.extend(base[(base.answer_type==kind)&(base.score==100)].head(2).question_id.tolist())
    for ids,name,title in [(case_ids,'06_errors','Zero-credit examples; deterministic selection'),(success,'07_successes','Full-credit examples; deterministic selection')]:
        fig,axes=plt.subplots(3,2,figsize=(13,16))
        for ax,qid in zip(axes.ravel(),ids):
            a=ann[qid];p=base[base.question_id==qid].iloc[0];path=image_path(data_root,a['image_id'])
            shutil.copy2(path,case_dir/path.name)
            with Image.open(path) as image:ax.imshow(image)
            refs=collections.Counter(x['answer'] for x in a['answers']).most_common(3)
            label=textwrap.fill(a['question'],58)+'\nPrediction: '+str(p.answer)+'\nReferences: '+', '.join(f'{v} ({n}/10)' for v,n in refs)
            ax.set_title(label,fontsize=9,loc='left');ax.axis('off')
        for ax in axes.ravel()[len(ids):]:ax.axis('off')
        fig.suptitle(title);fig.tight_layout(rect=(0,0,1,.97));save_figure(fig,figdir,name)
    cases=base[base.question_id.isin(case_ids)]
    cases.to_csv(out/'error_cases.csv',index=False)
    limitation={'split':'VQAv2 val2014','potential_training_overlap':True,'project_finetuning':False,
        'evidence':'https://github.com/salesforce/BLIP/blob/main/configs/vqa.yaml',
        'note':'Official training configuration contains vqa_val. Exact checkpoint lineage was not independently established. No unseen-test generalization claim.'}
    write_json(out/'limitations.json',limitation)
    text=f'''# 实验5：本地结果分析与PPT素材

固定 {len(manifest)} 题、{len(set(a['image_id'] for a in manifest))} 张图像，随机种子42。模型为Salesforce/blip-vqa-base，没有进行项目微调。全部预测来自本机实际运行。

**这是可能存在训练重叠的VQAv2 validation诊断，不是独立测试集成绩。** 官方BLIP配置含vqa_val；不能据此声称达到论文的未见测试性能。

## 准确率
{table.round(3).to_markdown(index=False)}

评分调用官方VQA评测器，只修改两处Python 2 print语法。保留数字、冠词、标点和缩写规范化，以及十位标注者留一法。每题0、30、60、90、100分，整体为平均值；不是单一标准答案的严格匹配率。

## 对照设计
greedy与beam3只改变束宽；blank_image保持问题和图像尺寸，用均匀灰图替换原图，检测图像证据缺失后的行为。灰图属于分布外干预，其下降不等于严格的因果贡献，也不是重新训练的纯文本基线。每种设置独立预热一个批次。

配对统计：
```json
{json.dumps(stats,ensure_ascii=False,indent=2)}
```
按图像聚类bootstrap，95%区间若包含0，不宣称稳定改善。解码消融没有针对这些题的答案调参。

## 性能
{timing.round(4).to_markdown()}
单位为秒/题的批量摊销成本。Generation仅计同步GPU生成；E2E还包括图像读取、预处理和解码，不包括下载、模型加载和评分。不能称为单请求响应延迟。

## 错误案例
{cases[['question_id','question','answer','majority']].to_markdown(index=False)}
按答案类别和题号固定选取前两个零分案例。多数票不一定是唯一真值，应结合per_question_results.csv中的十人答案分布解释。不能仅凭输出断言模型内部原因。

## 可用素材
figures/中7组PNG和SVG；case_images/保留12个左右可复核案例；predictions_*.jsonl是原始预测；submission_*.json为官方格式；per_question_results.csv是逐题评分；run_config、environment和image_hashes提供完整来源。

PPT建议：任务与架构 → 数据抽样 → 官方计分 → 总体与类别结果 → 解码成本 → 灰图对照 → 成功/失败案例 → 重叠限制与改进。引用见docs/REFERENCES.md。
'''
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
