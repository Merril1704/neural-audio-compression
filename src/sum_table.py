import pandas as pd
df = pd.read_csv('output/metrics.csv')
df['entropy_kbps'] = df['entropy_bps'] / 1000.0
agg = df.groupby(['method','bits']).agg(
			segsnr_mean=('segsnr_db','mean'), segsnr_std=('segsnr_db','std'),
			stoi_mean=('stoi','mean'), stoi_std=('stoi','std'),
			pesq_mean=('pesq','mean'), pesq_std=('pesq','std'),
			ent_mean=('entropy_kbps','mean'), ent_std=('entropy_kbps','std'),
).reset_index()
def pm(m,s):
		return (f"{m:.2f} ± {s:.2f}" if pd.notnull(m) and pd.notnull(s) else "n/a")
cols = ['Method','Bits','SegSNR (dB)','STOI','PESQ','Entropy (kbps)']
rows = []
for _,r in agg.iterrows():
			rows.append([
					r['method'], int(r['bits']),
					pm(r['segsnr_mean'], r['segsnr_std']),
					pm(r['stoi_mean'], r['stoi_std']),
					pm(r['pesq_mean'], r['pesq_std']),
					pm(r['ent_mean'], r['ent_std']),
			])
	# Print as Markdown
out = ['|'+'|'.join(cols)+'|', '|'+'|'.join(['---']*len(cols))+'|']
for row in rows:
		out.append('|'+'|'.join(map(str,row))+'|')
print('\n'.join(out))