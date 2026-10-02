"""Regenerate standalone SVG result charts from committed metrics; stdlib only."""
from pathlib import Path
import json
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'
ASPECTS = ['appearance', 'aroma', 'palate', 'taste', 'overall']


def chart(filename, title, subtitle, series, footnote):
    width, height = 1080, 570
    left, top, plot_width, plot_height = 90, 140, 900, 290
    ymax = .03
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)} {escape(footnote)}</desc>',
             '<rect width="1080" height="570" fill="#ffffff"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17283d} .muted{fill:#53657a}</style>']
    def text(x, y, content, size=16, **attributes):
        attrs=' '.join(f'{k.replace("_","-")}="{v}"' for k,v in attributes.items())
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {attrs}>{escape(content)}</text>')
    text(50, 48, title, 28, font_weight='bold')
    text(50, 78, subtitle, 17, class_='muted')
    for index, (name, values, color) in enumerate(series):
        x=50+index*330
        parts.append(f'<rect x="{x}" y="100" width="16" height="16" rx="3" fill="{color}"/>')
        text(x+24,114,name,16)
    for tick in [0,.01,.02,.03]:
        y=top+plot_height*(1-tick/ymax)
        parts.append(f'<line x1="{left}" x2="{left+plot_width}" y1="{y}" y2="{y}" stroke="#dde5ef"/>')
        text(left-14,y+5,f'{tick:.2f}',14,text_anchor='end')
    text(35,top+plot_height/2,'MSE',14,transform=f'rotate(-90 35 {top+plot_height/2})')
    cell=plot_width/len(ASPECTS);bar_width=48
    for j,aspect in enumerate(ASPECTS):
        center=left+cell*(j+.5)
        for i,(_,values,color) in enumerate(series):
            value=values[j];bar_height=value/ymax*plot_height
            x=center+(i-(len(series)-1)/2)*58-bar_width/2
            y=top+plot_height-bar_height
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="3"/>')
            text(x+bar_width/2,y-10,f'{value:.4f}',14,text_anchor='middle')
        text(center,top+plot_height+30,aspect.title(),16,text_anchor='middle')
    text(50,500,footnote,16,class_='muted')
    text(50,533,'Source: committed aggregate metrics. Lower MSE is better; targets are normalized to [0, 1].',14,class_='muted')
    parts.append('</svg>')
    (OUT/filename).write_text('\n'.join(parts).replace('class-=', 'class='),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((ROOT/'results/product-holdout/metrics.json').read_text())
    neural=json.loads((ROOT/'results/lora-smoke/metrics.json').read_text())
    chart('baseline-mse.svg','Unseen-product evaluation: text features improve the baseline',
          '20,000-record prefix | 3,559 test reviews | 76 held-out products',
          [('Training-set mean',[baseline['test_train_mean'][k]['mse'] for k in ASPECTS],'#a6b5ca'),
           ('Hashed BoW + Ridge',[baseline['test_ridge'][k]['mse'] for k in ASPECTS],'#147d92')],
          'Macro MSE: 0.018400 → 0.016149 (12.2% reduction on this sample).')
    chart('lora-smoke-mse.svg','LoRA smoke run: verified end-to-end inference',
          '2,000-record prefix | 1 CPU epoch | 46 test reviews',
          [('DistilBERT + LoRA',[neural['test'][k]['mse'] for k in ASPECTS],'#6856b8')],
          'Test macro MSE: 0.012316. Small-sample workflow check; not comparable to the ridge chart.')
    print('Wrote two SVG charts from recorded metrics:',OUT)


if __name__=='__main__':main()
