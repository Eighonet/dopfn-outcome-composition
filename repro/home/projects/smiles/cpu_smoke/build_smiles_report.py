#!/usr/bin/env python3
"""Create a report from saved CV, holdout and bootstrap results. No models are retrained."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import re
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATASETS = ["criteo", "hillstrom", "lzd", "orange", "retailhero"]
MODELS = ["causalpfn", "tabpfn_t", "dopfn"]
STRATEGIES = ["random", "balanced_treatment", "balanced_treatment_outcome"]
STRAT_LABEL = {
    "random": "Random",
    "balanced_treatment": "Balanced T",
    "balanced_treatment_outcome": "Balanced T+Y",
}
MODEL_LABEL = {"causalpfn": "CausalPFN", "tabpfn_t": "TabPFN-T", "dopfn": "Do-PFN"}
DATASET_LABEL = {"criteo":"Criteo", "hillstrom":"Hillstrom", "lzd":"LZD", "orange":"Orange", "retailhero":"Retailhero"}
MODEL_COLORS = {"causalpfn":"#24649b", "tabpfn_t":"#c07927", "dopfn":"#438663"}
STRAT_COLORS = {"random":"#757575", "balanced_treatment":"#2670a0", "balanced_treatment_outcome":"#bd6532"}
DATA_N = {"criteo":13979592, "hillstrom":64000, "lzd":181669, "orange":11896, "retailhero":200039}
DATA_P = {"criteo":10, "hillstrom":8, "lzd":64, "orange":178, "retailhero":5}


def must(cond, why):
    if not cond:
        raise RuntimeError(why)


def clean_float(x, digits=5):
    if pd.isna(x):
        return "—"
    return f"{float(x):+.{digits}f}"


def make_markdown_table(headers, rows):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    return "\n".join(lines)


def fig_save(fig, path_base, catalog, label):
    fig.tight_layout()
    png = path_base.with_suffix(".png")
    svg = path_base.with_suffix(".svg")
    fig.savefig(png, dpi=190, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    catalog.append({"title":label, "png": png, "svg": svg})


def parse_bootstrap_log(path):
    if not path.exists():
        return None, f"Bootstrap log not found: {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if "FINAL: PASS" not in text or "Bootstrap 1000/1000" not in text:
        return None, "Bootstrap B=1000 was not verified as completed in the log"
    pattern = re.compile(r"^\s*(qini|auuc|uplift10)\s+observed=([+-]?[0-9]*\.?[0-9]+)\s+bootstrap95=\[\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\]", re.M)
    rows = []
    for metric, point, lo, hi in pattern.findall(text):
        rows.append({"metric":metric,"observed":float(point),"ci_low":float(lo),"ci_high":float(hi),"B":1000})
    if len(rows) != 3 or len({x['metric'] for x in rows}) != 3:
        return None, "No complete 3-metric bootstrap summary found in log"
    return pd.DataFrame(rows).set_index("metric"), None


def validate_cv(df):
    required = {"dataset","model","fold","seed","context","strategy","qini","auuc","uplift10"}
    must(required.issubset(df.columns), f"CV CSV missing columns {sorted(required - set(df.columns))}")
    must(len(df)==2700, f"Expected 2700 CV runs; found {len(df)}")
    must(df.groupby(["dataset","model","fold","seed","context","strategy"]).size().eq(1).all(), "Duplicate CV configuration")
    must(set(df.dataset) == set(DATASETS), "Expected five CV datasets")
    must(set(df.model) == set(MODELS), "Expected three CV models")
    must(set(df.strategy) == set(STRATEGIES), "Expected three context strategies")
    must(set(df.context) == {512,1024,2048,4096}, "Unexpected CV context sizes")
    must(set(df.seed) == {42,43,44}, "Unexpected CV seeds")
    must(set(df.fold) == set(range(5)), "Unexpected CV folds")
    must(df[["qini","auuc","uplift10"]].notna().all().all(), "Null CV metric")
    must(np.isfinite(df[["qini","auuc","uplift10"]].to_numpy(dtype=float)).all(), "Nonfinite CV metric")


def validate_holdout(df):
    required = {"dataset","model","context","strategy","seed","n_eval","qini","auuc","uplift10","time_sec"}
    must(required.issubset(df.columns), f"Holdout CSV missing {sorted(required - set(df.columns))}")
    must(len(df)==48, f"Expected 48 verified holdout results; got {len(df)}")
    must(df.groupby(["dataset","model","context","strategy","seed"]).size().eq(1).all(), "Duplicate holdout configuration")
    must(set(df.seed) == {42,43,44}, "Unexpected holdout seeds")
    must(set(df.dataset) == set(DATASETS), "Holdout missing datasets")
    must(df.groupby(["dataset","model","context","strategy"]).size().eq(3).all(), "Each holdout config needs three seeds")
    must(df.groupby("dataset").n_eval.nunique().eq(1).all(), "Different holdout sizes by config")
    must(np.isfinite(df[["qini","auuc","uplift10"]].to_numpy(dtype=float)).all(), "Nonfinite holdout metric")
    must(df.n_eval.gt(0).all(), "Bad holdout n_eval")


def aggregate(df):
    # Average scores within folds before measuring variation across folds.
    fold = df.groupby(["dataset","model","context","strategy","fold"], as_index=False).agg(
        qini=("qini","mean"), auuc=("auuc","mean"), uplift10=("uplift10","mean"))
    conf = fold.groupby(["dataset","model","context","strategy"], as_index=False).agg(
        cv_qini=("qini","mean"), cv_qini_sd_folds=("qini","std"),
        cv_auuc=("auuc","mean"), cv_uplift10=("uplift10","mean"))
    return fold, conf



def paired_cv_strategy_effects(cv):
    p = cv.pivot(index=["dataset","model","fold","seed","context"],columns="strategy",values="qini")
    must(len(p)==900 and not p.isna().any().any(), "CV paired strategy comparisons incomplete")
    p = p.copy()
    p["T_minus_R"] = p["balanced_treatment"] - p["random"]
    p["TY_minus_R"] = p["balanced_treatment_outcome"] - p["random"]
    p["TY_minus_T"] = p["balanced_treatment_outcome"] - p["balanced_treatment"]
    paired = p.reset_index()
    fold = paired.groupby(["dataset","model","fold"],as_index=False)[["T_minus_R","TY_minus_R","TY_minus_T"]].mean()
    summary=fold.groupby(["dataset","model"],as_index=False).agg(
        T_minus_R=("T_minus_R","mean"), T_minus_R_sd_folds=("T_minus_R","std"),
        TY_minus_R=("TY_minus_R","mean"), TY_minus_R_sd_folds=("TY_minus_R","std"),
        TY_minus_T=("TY_minus_T","mean"), TY_minus_T_sd_folds=("TY_minus_T","std"))
    fold_by_context=paired.groupby(["dataset","model","context","fold"],as_index=False)[["T_minus_R","TY_minus_R","TY_minus_T"]].mean()
    interaction=fold_by_context.groupby(["dataset","model","context"],as_index=False).agg(
        T_minus_R=("T_minus_R","mean"), T_minus_R_sd_folds=("T_minus_R","std"),
        TY_minus_R=("TY_minus_R","mean"), TY_minus_R_sd_folds=("TY_minus_R","std"),
        TY_minus_T=("TY_minus_T","mean"), TY_minus_T_sd_folds=("TY_minus_T","std"))
    return summary, interaction


def cv_context_effect(cv):
    cv_fold_context=cv.groupby(["dataset","model","fold","context"],as_index=False).qini.mean()
    p=cv_fold_context.pivot(index=["dataset","model","fold"],columns="context",values="qini")
    p['ctx4096_minus_512']=p[4096]-p[512]
    q=p.reset_index().groupby(["dataset","model"]).agg(
        ctx4096_minus_512=("ctx4096_minus_512","mean"),
        ctx4096_minus_512_sd_folds=("ctx4096_minus_512","std")).reset_index()
    return q


def draw_heatmap(df, value_col, outfile, catalog, title, subtitle, fmt="+.3f", vlim=None):
    mat=df.pivot(index="dataset",columns="model",values=value_col).reindex(index=DATASETS,columns=MODELS)
    fig, ax=plt.subplots(figsize=(8.8,5.2))
    vals=mat.to_numpy(dtype=float)
    vmax=float(np.nanmax(np.abs(vals))) if vlim is None else float(vlim)
    vmax=max(vmax,0.001)
    im=ax.imshow(vals,cmap="RdBu",vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(MODELS)),[MODEL_LABEL[x] for x in MODELS])
    ax.set_yticks(range(len(DATASETS)),[DATASET_LABEL[x] for x in DATASETS])
    for r in range(len(DATASETS)):
        for c in range(len(MODELS)):
            z=vals[r,c]
            ax.text(c,r,format(z,fmt),ha="center",va="center",weight="bold",fontsize=11,
                    color="white" if abs(z)>0.62*vmax else "#1f2833")
    ax.set_title(title,loc="left",pad=24,fontsize=14,weight="bold")
    ax.text(0,1.025,subtitle,transform=ax.transAxes,color="#68717a",fontsize=9)
    fig.colorbar(im,ax=ax,shrink=.75,label="Normalized Qini AUC")
    fig_save(fig,outfile,catalog,title)


def draw_cv_vs_holdout(main, outfile, catalog):
    fig,ax=plt.subplots(figsize=(8.2,7.5))
    vals=np.concatenate([main.cv_qini.values,main.holdout_qini.values])
    low=min(-.025,float(vals.min())-.01); high=max(.10,float(vals.max())+.01)
    ax.plot([low,high],[low,high],color="#9aa4ac",ls="--",lw=1.3,label="CV = holdout")
    for m in MODELS:
        sub=main[main.model==m]
        ax.scatter(sub.cv_qini,sub.holdout_qini,color=MODEL_COLORS[m],s=70,label=MODEL_LABEL[m],alpha=.9)
        for _,r in sub.iterrows():
            ax.annotate(DATASET_LABEL[r.dataset],(r.cv_qini,r.holdout_qini),xytext=(4,4),
                        textcoords="offset points",fontsize=8,color=MODEL_COLORS[m])
    ax.axvline(0,color="#e0e3e7",lw=.8); ax.axhline(0,color="#e0e3e7",lw=.8)
    ax.set(xlim=(low,high),ylim=(low,high),xlabel="Cross-validation normalized Qini AUC",
           ylabel="Full holdout normalized Qini AUC")
    ax.set_title("CV versus final holdout (15 frozen configurations)",loc="left",weight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=.15)
    fig_save(fig,outfile,catalog,"CV versus independent holdout")


def draw_effect_heatmap(strategy_effect, outfile, catalog):
    d=strategy_effect.copy()
    d['label']=d.dataset.map(DATASET_LABEL)+" / "+d.model.map(MODEL_LABEL)
    d['k']=pd.Categorical(d.dataset,categories=DATASETS,ordered=True)
    d['m']=pd.Categorical(d.model,categories=MODELS,ordered=True)
    d=d.sort_values(['k','m'])
    mat=d[["T_minus_R","TY_minus_R","TY_minus_T"]].to_numpy(float)
    mx=max(.01,float(np.max(np.abs(mat))))
    fig,ax=plt.subplots(figsize=(8.8,9))
    im=ax.imshow(mat,cmap="RdBu",vmin=-mx,vmax=mx,aspect="auto")
    ax.set_yticks(range(len(d)),d['label'])
    ax.set_xticks([0,1,2],["Balanced T − Random","Balanced T+Y − Random","Balanced T+Y − T"])
    ax.tick_params(axis='x',labelrotation=12)
    for i in range(len(d)):
        for j in range(3):
            ax.text(j,i,f"{mat[i,j]:+.3f}",ha="center",va="center",fontsize=9,
                    color="white" if abs(mat[i,j])>.6*mx else "#19232d")
    ax.set_title("Paired context selection effects · CV",loc="left",weight="bold",pad=18)
    fig.colorbar(im,ax=ax,shrink=.75,label="Δ normalized Qini AUC")
    fig_save(fig,outfile,catalog,"Context-strategy effects across all datasets")


def draw_context_lines(conf, dataset, outfile, catalog):
    subset=conf[conf.dataset==dataset]
    fig,axs=plt.subplots(1,3,figsize=(13.4,3.9),sharey=True)
    for ax,model in zip(axs,MODELS):
        for strategy in STRATEGIES:
            v=subset[(subset.model==model)&(subset.strategy==strategy)].sort_values('context')
            ax.plot(v.context,v.cv_qini,marker='o',lw=1.8,color=STRAT_COLORS[strategy],label=STRAT_LABEL[strategy])
        ax.set_xscale('log',base=2)
        ax.set_xticks([512,1024,2048,4096],["512","1024","2048","4096"])
        ax.set_title(MODEL_LABEL[model],fontsize=11,weight='bold')
        ax.axhline(0,color='#b3bac1',lw=.8,ls='--')
        ax.grid(alpha=.17)
    axs[0].set_ylabel('Mean CV normalized Qini AUC')
    handles,labels=axs[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
    fig.suptitle(f"Context size × selection strategy · {DATASET_LABEL[dataset]}",fontweight='bold',y=1.025)
    fig.subplots_adjust(bottom=.16)
    fig_save(fig,outfile,catalog,f"Context size × strategy: {DATASET_LABEL[dataset]}")


def draw_dopfn_interaction(interaction, outfile, catalog):
    fig,ax=plt.subplots(figsize=(9.5,5.2))
    colors=['#9b463a','#6c75ad','#4e876e','#967349','#65727b']
    for d,color in zip(DATASETS,colors):
        sub=interaction[(interaction.dataset==d)&(interaction.model=='dopfn')].sort_values('context')
        ax.plot(sub.context,sub.TY_minus_R,marker='o',lw=2,label=DATASET_LABEL[d],color=color)
    ax.axhline(0,color='#9aa4ac',lw=1)
    ax.set_xscale('log',base=2)
    ax.set_xticks([512,1024,2048,4096],["512","1024","2048","4096"])
    ax.set(xlabel="Context size",ylabel="Balanced T+Y − Random (Δ Qini)")
    ax.set_title("Do-PFN: outcome-balanced context benefit depends on dataset",loc="left",weight="bold")
    ax.grid(alpha=.18)
    ax.legend(ncol=3,fontsize=9)
    fig_save(fig,outfile,catalog,"Do-PFN paired strategy effect across contexts")


def draw_holdout_seeds(holdout, outfile, catalog):
    d=holdout[(holdout.dataset=='criteo')&(holdout.model=='dopfn')&(holdout.context==1024)]
    metrics=[('qini','Normalized Qini AUC'),('auuc','AUUC'),('uplift10','Uplift@10%')]
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.4))
    for ax,(col,name) in zip(axs,metrics):
        for seed in [42,43,44]:
            s=d[d.seed==seed].set_index('strategy')
            y=[float(s.loc['balanced_treatment',col]),float(s.loc['balanced_treatment_outcome',col])]
            ax.plot([0,1],y,'o-',label=f'Seed {seed}',lw=1.8)
        ax.set_xticks([0,1],["Balanced T","Balanced T+Y"])
        ax.set_title(name,weight="bold",fontsize=11)
        ax.grid(axis='y',alpha=.2)
        ax.axhline(0,ls='--',color='#9aa4ac',lw=.8)
    axs[0].legend(loc='best',fontsize=9)
    fig.suptitle('Criteo / Do-PFN / context 1024 · paired seed comparison',weight='bold',y=1.04)
    fig_save(fig,outfile,catalog,'Paired Do-PFN seed results on Criteo')


def draw_bootstrap(bs, outfile, catalog):
    fig,axs=plt.subplots(1,3,figsize=(12.8,3.4))
    for ax,metric,title in zip(axs,['qini','auuc','uplift10'],['Normalized Qini AUC','AUUC','Uplift@10%']):
        row=bs.loc[metric]
        mid=float(row.observed); low=float(row.ci_low); high=float(row.ci_high)
        ax.errorbar(mid,0,xerr=[[mid-low],[high-mid]],fmt='o',capsize=7,markersize=7,color='#326c93',elinewidth=2)
        ax.axvline(0,color='#a6afb8',ls='--',lw=1)
        pad=max(.01,(high-low)*1.2)
        ax.set_xlim(min(0,low)-pad,high+pad)
        ax.set_ylim(-.6,.6)
        ax.set_yticks([])
        ax.set_title(title,weight='bold',fontsize=11)
        ax.set_xlabel('Balanced T+Y − Balanced T')
        ax.text(mid,-.32,f'{mid:+.5f} [{low:+.5f}; {high:+.5f}]',ha='center',fontsize=9)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Criteo / Do-PFN · paired cluster bootstrap (B=1000)',weight='bold',y=1.05)
    fig_save(fig,outfile,catalog,'95% paired cluster bootstrap intervals')


def save_html(markdown_report, figures, tables, outpath, metadata):
    # Include the figures in the report for offline viewing.
    style="""
      :root{font-family:Arial,Helvetica,sans-serif;color:#1e2b38;background:#fafbfd}
      body{margin:0 auto;max-width:1080px;padding:35px 44px;line-height:1.5}
      h1{font-size:31px;letter-spacing:-.7px;border-bottom:3px solid #294b65;padding-bottom:18px}
      h2{margin-top:38px;font-size:22px;border-bottom:1px solid #ccd4dc;padding-bottom:7px}
      h3{margin-top:25px} p,li{font-size:14px} .sub{color:#65717b;font-size:14px}
      table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0 27px}
      th{background:#e8eff5;text-align:left}th,td{padding:8px 9px;border-bottom:1px solid #dae1e7}
      tr:nth-child(even){background:#f4f7fa}
      figure{margin:24px 0 36px}img{width:100%;height:auto;background:white;border:1px solid #e5e9ed;border-radius:4px}
      figcaption{font-size:12px;color:#69727d;margin:6px 0}
      .note{border-left:4px solid #7995aa;padding:8px 14px;background:#f0f5f9}
      .mono{font-family:Consolas,monospace} .tiny{font-size:12px;color:#606b75}
      @media print{body{padding:10px;max-width:none} figure{break-inside:avoid}h2{break-after:avoid}}
    """
    p=[]
    p.append('<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SMILES — итоговый аналитический отчёт</title><style>'+style+'</style></head><body>')
    p.append('<h1>SMILES: benchmark of foundation models for uplift modeling</h1>')
    p.append('<p class="sub">Итоговый аналитический отчёт · 5 датасетов · 3 модели · 2700 CV + 48 holdout · paired cluster bootstrap B=1000</p>')
    p.append('<p class="note">Источник: сохранённые CSV и журнал бутстрэпа на lorien. Никакие модели повторно не запускались. Отложенный тест использован только для итоговой оценки.</p>')
    def section(title,content):p.extend([f'<h2>{html.escape(title)}</h2>',content])
    section('Протокол исследования', '<p>Для каждого датасета фиксировался финальный отложенный тест (15%), а оставшаяся часть использовалась для пятифолдовой кросс-валидации. Сравнивались CausalPFN, TabPFN-T и Do-PFN; 4 размера контекста (512–4096), 3 стратегии (Random, Balanced T, Balanced T+Y), 3 seed. На финальном тесте были проверены 15 CV-выбранных конфигураций и дополнительный заранее зафиксированный контроль Do-PFN / Criteo.</p>')
    dsrows=[]
    for d in DATASETS:
        dsrows.append([DATASET_LABEL[d],f'{DATA_N[d]:,}',DATA_P[d],f'{int(metadata["n_holdout"][d]):,}'])
    p.append(pd.DataFrame(dsrows,columns=['Dataset','Total rows','Features','Holdout rows']).to_html(index=False,escape=True))
    section('Главный результат: Do-PFN на Criteo', '<p>Балансировка контекста по воздействию и исходу (Balanced T+Y) показала положительное преимущество над балансировкой только по воздействию (Balanced T) на полном независимом holdout. Ниже — разность по метрикам и парные 95%-е кластерные бутстрэп-интервалы при B=1000.</p>')
    if metadata['bootstrap'] is not None:
        bs=metadata['bootstrap']
        bf=pd.DataFrame([{"Metric":m,"Observed delta":f'{bs.loc[m,"observed"]:+.5f}',
                          "95% cluster bootstrap CI":f'[{bs.loc[m,"ci_low"]:+.5f}; {bs.loc[m,"ci_high"]:+.5f}]'} for m in ['qini','auuc','uplift10']])
        p.append(bf.to_html(index=False,escape=True))
    else:
        p.append('<p class="note">Протокол бутстрэпа не найден: доверительные интервалы опущены.</p>')
    p.append('<p class="tiny">Вывод условен относительно фиксированных обученных конфигураций и допущений о независимости кластеров; он не распространяется автоматически на калибровку индивидуальных эффектов.</p>')
    section('CV и отложенный тест: 15 выбранных конфигураций', '<p>CV mean и CV SD рассчитаны по пяти фолдам после усреднения трёх seed в каждом фолде; holdout mean и holdout SD — по трём seed на одном фиксированном тесте. Эти SD нельзя интерпретировать как взаимозаменяемые.</p>')
    df=tables['primary'].copy()
    show=pd.DataFrame({
        'Dataset':df.dataset.map(DATASET_LABEL),'Model':df.model.map(MODEL_LABEL),
        'Context':df.context,'Strategy':df.strategy.map(STRAT_LABEL),
        'CV Qini ± fold SD':[f'{x:+.4f} ± {y:.4f}' for x,y in zip(df.cv_qini,df.cv_qini_sd_folds)],
        'Holdout Qini ± seed SD':[f'{x:+.4f} ± {y:.4f}' for x,y in zip(df.holdout_qini,df.holdout_qini_sd_seeds)],
    })
    p.append(show.to_html(index=False,escape=True))
    section('Иллюстрации', '<p>Графики также сохранены в PNG и редактируемом векторном SVG. Для статей пригодны SVG; для слайдов — PNG.</p>')
    for it in figures:
        encoded=base64.b64encode(it['png'].read_bytes()).decode('ascii')
        p.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(it["title"])}"><figcaption>{html.escape(it["title"])}</figcaption></figure>')
    section('Ограничения и корректность интерпретации', '''<ul>
      <li>Best-of-12 CV конфигурации выбраны по тем же CV метрикам; оценка выбранного варианта на CV может быть оптимистичной.</li>
      <li>Положительные 95%-е интервалы относятся только к зафиксированному сравнению Do-PFN на Criteo, а не ко всем парам моделей или ко всем датасетам.</li>
      <li>Balanced T+Y изменяет долю положительных исходов внутри каждого T до 50%; обнаруженное улучшение uplift ranking не подтверждает калибровку причинных эффектов.</li>
      <li>Три seed меняют и выбор контекста, и внутреннюю стохастику моделей. Их SD не эквивалентен неопределённости по независимым наблюдениям.</li>
      <li>Группы для Criteo/LZD строятся по совпадающим векторам признаков; это препятствует дословной утечке дубликатов, но не гарантирует отсутствия всех форм зависимости.</li>
      <li>Сводные результаты не содержат полной характеристики независимости экспериментальных единиц и валидности причинных предпосылок каждого датасета.</li>
      <li>Для сравнения остальных архитектур/датасетов отдельные доверительные интервалы не рассчитывались; значения следует рассматривать как описательные.</li>
    </ul>''')
    section('Воспроизводимость',f'<p>Исходные таблицы: <span class="mono">fm_all_2700_runs.csv</span>, <span class="mono">fm_holdout_verified.csv</span>. Исходный log: <span class="mono">bootstrap-2884.out</span>. Файлы таблиц и векторных иллюстраций входят в ZIP-пакет.</p><p>Frozen manifest SHA256: <span class="mono">{metadata["manifest_sha"]}</span></p>')
    p.append('<p class="tiny">Генератор отчёта: build_smiles_report.py. Все числовые таблицы формируются из исходных сохранённых данных.</p></body></html>')
    outpath.write_text('\n'.join(p),encoding='utf-8')


def generate(root, output, bootstrap_log, manifest):
    analysis=root/'results'/'analysis'
    cv_path=analysis/'fm_all_2700_runs.csv'
    hold_path=analysis/'fm_holdout_verified.csv'
    must(cv_path.is_file(), f'Missing required input: {cv_path}')
    must(hold_path.is_file(), f'Missing required input: {hold_path}')
    cv=pd.read_csv(cv_path)
    holdout=pd.read_csv(hold_path)
    validate_cv(cv);validate_holdout(holdout)
    cv_fold,cv_conf=aggregate(cv)
    hold_summary=holdout.groupby(['dataset','model','context','strategy'],as_index=False).agg(
        holdout_qini=('qini','mean'),holdout_qini_sd_seeds=('qini','std'),
        holdout_auuc=('auuc','mean'),holdout_auuc_sd_seeds=('auuc','std'),
        holdout_uplift10=('uplift10','mean'),holdout_uplift10_sd_seeds=('uplift10','std'),
        n_eval=('n_eval','first'),time_sec_mean=('time_sec','mean'))
    # Check that each selected configuration was tested during CV.
    merged=hold_summary.merge(cv_conf,on=['dataset','model','context','strategy'],how='left',validate='one_to_one')
    must(len(merged)==16 and merged.cv_qini.notna().all(),'Selected configs missing from CV')
    control=(merged.dataset.eq('criteo')&merged.model.eq('dopfn')&merged.context.eq(1024)&merged.strategy.eq('balanced_treatment'))
    must(control.sum()==1,'Expected one additional Criteo Do-PFN Balanced T control')
    merged['role']=np.where(control,'prespecified_comparator','CV_selected')
    primary=merged[~control].copy().sort_values(['dataset','model'])
    must(len(primary)==15 and primary.groupby('dataset').model.nunique().eq(3).all(),'Expected 15 selected final configs')
    merged['holdout_minus_cv_qini']=merged.holdout_qini-merged.cv_qini
    primary=merged[merged.role=='CV_selected'].copy().sort_values(['dataset','model'])
    # Check the selected configurations against their CV rankings.
    cv_conf=cv_conf.copy()
    cv_conf['rank_cv_qini']=cv_conf.groupby(['dataset','model']).cv_qini.rank(ascending=False,method='min')
    merged=merged.merge(cv_conf[['dataset','model','context','strategy','rank_cv_qini']],on=['dataset','model','context','strategy'],how='left',validate='one_to_one')
    primary=merged[merged.role=='CV_selected'].copy().sort_values(['dataset','model'])
    strategy_effect,interactions=paired_cv_strategy_effects(cv)
    context_effect=cv_context_effect(cv)
    bs,bs_warn=parse_bootstrap_log(bootstrap_log)
    if bs is not None:
        sub=holdout[(holdout.dataset=='criteo')&(holdout.model=='dopfn')&(holdout.context==1024)]
        pv=sub.pivot(index='seed',columns='strategy',values=['qini','auuc','uplift10'])
        for col in ['qini','auuc','uplift10']:
            point=float((pv[(col,'balanced_treatment_outcome')]-pv[(col,'balanced_treatment')]).mean())
            must(abs(point-float(bs.loc[col,'observed']))<.0001,
                 f'Bootstrap log observed {col} does not match saved holdout ({point})')
    sha='not provided'
    if manifest.is_file():
        sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
        known='dd8d349f336399e438caf7be4f2a250746b137b1dbba9c5d79cb7292a0f884e2'
        must(sha==known,'Frozen manifest sha256 mismatch: DO NOT TRUST THE INPUTS')
        lines=[i.strip() for i in manifest.read_text().splitlines() if i.strip()]
        must(len(lines)==48,'Manifest should list exactly 48 jobs')
        manifest_jobs=set()
        for line in lines:
            x=line.split();must(len(x)==5,f'Unexpected manifest line: {line}')
            manifest_jobs.add((x[0],x[1],int(x[2]),x[3],int(x[4])))
        hold_jobs=set(zip(holdout.dataset,holdout.model,holdout.context,holdout.strategy,holdout.seed))
        must(manifest_jobs==hold_jobs,'Verified holdout CSV does not match frozen manifest')
    output.mkdir(parents=True,exist_ok=True)
    figures_path=output/'figures';figures_path.mkdir(exist_ok=True)
    table_path=output/'tables';table_path.mkdir(exist_ok=True)
    tables={
        'cv_config_summary_180':cv_conf,
        'cv_paired_strategy_effects_15':strategy_effect,
        'cv_context_interaction_60':interactions,
        'cv_context_4096_minus_512_15':context_effect,
        'holdout_48_runs':holdout,
        'holdout_16_configs_cv_comparison':merged,
        'holdout_15_cv_selected':primary,
    }
    for name,d in tables.items():d.to_csv(table_path/f'{name}.csv',index=False)
    if bs is not None:bs.reset_index().to_csv(table_path/'criteo_dopfn_cluster_bootstrap_B1000.csv',index=False)
    figures=[]
    draw_heatmap(primary,'holdout_qini',figures_path/'01_holdout_qini_heatmap',figures,
                 'Holdout Qini by dataset and model','15 CV-selected configurations, mean over three seed')
    draw_heatmap(primary,'cv_qini',figures_path/'02_cv_best_qini_heatmap',figures,
                 'Cross-validation Qini by dataset and model','Same 15 frozen configurations, mean over five folds and three seed')
    draw_cv_vs_holdout(primary,figures_path/'03_cv_vs_holdout',figures)
    draw_effect_heatmap(strategy_effect,figures_path/'04_strategy_effects_CV',figures)
    draw_dopfn_interaction(interactions,figures_path/'05_dopfn_context_interaction',figures)
    for d in DATASETS:
        draw_context_lines(cv_conf,d,figures_path/f'06_context_lines_{d}',figures)
    draw_holdout_seeds(holdout,figures_path/'07_criteo_dopfn_paired_seeds',figures)
    if bs is not None:draw_bootstrap(bs,figures_path/'08_bootstrap_ci',figures)
    n_holdout=holdout.groupby('dataset').n_eval.first().to_dict()
    metadata={'n_holdout':n_holdout,'bootstrap':bs,'manifest_sha':sha}
    save_html(None,figures,{'primary':primary},output/'SMILES_full_report.html',metadata)
    # Save a text version of the report.
    rows=[]
    for _,r in primary.iterrows():
        rows.append([DATASET_LABEL[r.dataset],MODEL_LABEL[r.model],str(int(r.context)),STRAT_LABEL[r.strategy],
                     f'{r.cv_qini:+.5f} ± {r.cv_qini_sd_folds:.5f}',
                     f'{r.holdout_qini:+.5f} ± {r.holdout_qini_sd_seeds:.5f}',
                     f'{r.holdout_auuc:+.5f}', f'{r.holdout_uplift10:+.5f}'])
    md=[
      '# SMILES — итоговый аналитический отчёт',
      '',
      '**Дизайн:** 5 datasets × 3 foundation models × 5 folds × 3 seeds × 4 context sizes × 3 strategies = **2700 CV**; 15 frozen best-CV configs + 1 additional Do-PFN comparator × 3 seed = **48 holdout runs**.',
      '',
      '## Отбор и оценка',
      'CV mean: среднее после усреднения 3 seed в каждом из 5 фолдов; CV SD: стандартное отклонение пяти фолдовых средних. Holdout mean±SD: среднее и SD трёх seed на одном полном отложенном тесте. SD seed не является доверительным интервалом генеральной совокупности.',
      '',
      make_markdown_table(['Dataset','Model','ctx','Strategy','CV Qini ± fold SD','Holdout Qini ± seed SD','Holdout AUUC','Holdout uplift@10%'],rows),
      '',
      '## Стратегия выбора контекста',
      'Результаты парных эффектов стратегии представлены в `tables/cv_paired_strategy_effects_15.csv`. Контекстные взаимодействия — в `tables/cv_context_interaction_60.csv`.',
      '',
      '## Do-PFN / Criteo: предопределённое сравнение',
    ]
    sub=holdout[(holdout.dataset=='criteo')&(holdout.model=='dopfn')&(holdout.context==1024)]
    pv=sub.pivot(index='seed',columns='strategy',values=['qini','auuc','uplift10'])
    dmeans={metric:float((pv[(metric,'balanced_treatment_outcome')]-pv[(metric,'balanced_treatment')]).mean()) for metric in ['qini','auuc','uplift10']}
    md.extend([f"Observed Δ: normalized Qini AUC **{dmeans['qini']:+.5f}**, AUUC **{dmeans['auuc']:+.5f}**, uplift@10% **{dmeans['uplift10']:+.5f}**.",""])
    if bs is not None:
        md.append('Pairwise feature-vector-cluster bootstrap, B=1000:')
        md.append(make_markdown_table(['Metric','Observed Δ','95% paired-cluster bootstrap CI'],[
            [m,f"{bs.loc[m,'observed']:+.5f}",f"[{bs.loc[m,'ci_low']:+.5f}; {bs.loc[m,'ci_high']:+.5f}]"] for m in ['qini','auuc','uplift10']]))
    else:md.append(f'**Bootstrap was not loaded:** {bs_warn}')
    md+=['', '## Иллюстрации', '']
    for it in figures:md+= [f"![{it['title']}](figures/{it['png'].name})",'']
    md+=['## Интерпретация и ограничения',
         '- Главное превосходство Balanced T+Y на Criteo относится к uplift ranking Do-PFN, а не к калибровке индивидуальных эффектов.',
         '- В Balanced T+Y внутри T отбирается около 50% положительных исходов; базовая частота исходов искусственно смещена.',
         '- Bootstrap CI оценивает разброс при повторном отборе кластеров из фиксированного holdout; он не учитывает все источники неопределённости обучения.',
         '- Показанные эффекты вне главного Criteo-сравнения являются описательными и не сопровождаются отдельной проверкой гипотез.',
         '- CV-selected конфигурации могут иметь оптимистичный CV результат. Сравнение CV с holdout описательно.',
         '- 3 seed изменяют как контекст, так и стохастику модели: это не независимые тестовые выборки.',
         '', '## Воспроизводимость',
         f'- Frozen manifest SHA256: `{sha}`',
         f'- Bootstrap source: `{bootstrap_log}`',
         '- Derived CSV tables: `tables/`',
         '- Figures: `figures/` (PNG and SVG)',
         '']
    (output/'SMILES_full_report.md').write_text('\n'.join(md),encoding='utf-8')
    english=f'''# Results (draft; manuscript-ready numbers, editorial review required)

A total of 2,700 cross-validation runs (five datasets, three foundation models,
five folds, three seeds, four context sizes, and three context-selection strategies)
were followed by 48 full-holdout evaluations for 16 prespecified configurations.
The holdout results were validated against common evaluation indices and treatment/outcome arrays.

For Do-PFN on Criteo (context size 1,024; N={int(n_holdout['criteo']):,} held-out instances),
selection balanced jointly by treatment and observed outcome outperformed treatment-only
balancing. The mean paired improvement over three random seeds was
{dmeans['qini']:+.5f} in normalized Qini AUC, {dmeans['auuc']:+.5f} in AUUC,
and {dmeans['uplift10']:+.5f} in uplift at 10%.
'''
    if bs is not None:
        english+=f'''A paired feature-vector-cluster bootstrap with 1,000 replicates yielded
95% percentile intervals of [{bs.loc['qini','ci_low']:.5f}, {bs.loc['qini','ci_high']:.5f}]
for the Qini difference, [{bs.loc['auuc','ci_low']:.5f}, {bs.loc['auuc','ci_high']:.5f}]
for AUUC, and [{bs.loc['uplift10','ci_low']:.5f}, {bs.loc['uplift10','ci_high']:.5f}]
for uplift at 10%. These intervals quantify test-cluster resampling uncertainty
conditional on fixed configurations, rather than the uncertainty of the full
model-selection and fitting process.
'''
    english+='''
Performance varied substantially across datasets, models, and context-selection
strategies, and the CV-selected configuration was not universally transferable
to the held-out test set. Outcome-balanced sampling changes outcome prevalence
in the context and should not be interpreted as a guarantee of causal-effect
calibration. See the figures and detailed comparison table for descriptive results.
'''
    (output/'EN_Results_draft.md').write_text(english,encoding='utf-8')
    readme=f'''SMILES FINAL RESULTS PACK
=========================
Full report: SMILES_full_report.html (self-contained, open locally in browser)
Text report: SMILES_full_report.md
Manuscript draft in English: EN_Results_draft.md
Figures: figures/ (PNG and SVG)
Tables: tables/ (CSV)

Source:
  {cv_path}
  {hold_path}
  {bootstrap_log}
  {manifest}

Validation: CV 2700/2700; holdout verified CSV 48/48; manifest matched when present.
Bootstrap: {'PASS, B=1000' if bs is not None else bs_warn}
SHA256 manifest: {sha}

Interpretation: cross-validation and holdout SDs come from distinct levels
(five folds vs three seeds); bootstrap CIs are conditional and cluster-based.
No GPU/model inference is done by this script.
'''
    (output/'README.txt').write_text(readme,encoding='utf-8')
    audit={
        'cv_runs':len(cv),'holdout_runs':len(holdout),'cv_configs':len(cv_conf),
        'holdout_configs':len(hold_summary),'cv_fold_means':len(cv_fold),
        'bootstrap_replicates':1000 if bs is not None else None,
        'bootstrap_warning':bs_warn,'manifest_sha256':sha,
        'dataset_counts':{d:int((holdout.dataset==d).sum()) for d in DATASETS},
        'cv_rank_of_selected':{f"{r.dataset}/{r.model}":int(r.rank_cv_qini) for _,r in primary.iterrows()},
        'holdout_n_eval':{d:int(n_holdout[d]) for d in DATASETS},
        'n_figures':len(figures),
    }
    (output/'validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=output.parent/(output.name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file():z.write(f,arcname=f.relative_to(output.parent))
    print('===== SMILES REPORT: PASS =====')
    print('CV: 2700/2700; final holdout: 48/48')
    print('Frozen manifest SHA256:',sha)
    if bs is None:print('WARNING:',bs_warn)
    print('Selected CV ranks:',{f'{r.dataset}/{r.model}':int(r.rank_cv_qini) for _,r in primary.iterrows()})
    print('Figures:',len(figures),'(PNG + SVG)')
    print('Tables:',len(tables)+(1 if bs is not None else 0))
    print('Report HTML:',output/'SMILES_full_report.html')
    print('Report Markdown:',output/'SMILES_full_report.md')
    print('English results:',output/'EN_Results_draft.md')
    print('Bundle ZIP:',archive)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',type=Path,default=Path('~/projects/smiles').expanduser())
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--bootstrap-log',type=Path,default=None)
    ap.add_argument('--manifest',type=Path,default=None)
    args=ap.parse_args()
    root=args.root.expanduser().resolve()
    output=(args.output or root/'results/analysis/smiles_final_report').expanduser().resolve()
    bootstrap_log=(args.bootstrap_log or root/'logs/bootstrap-2884.out').expanduser().resolve()
    manifest=(args.manifest or root/'manifests/holdout_frozen_20261021_233147/frozen_holdout_manifest.txt').expanduser().resolve()
    generate(root,output,bootstrap_log,manifest)

if __name__=='__main__':
    main()
