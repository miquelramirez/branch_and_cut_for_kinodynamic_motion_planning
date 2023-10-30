#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import os

#sys.path += ['notebooks']
#os.chdir('notebooks')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_support import *
from plots import *

# IMPORTANT: need to install package `cm-super-minimal` and `dvipng` for matplotlib Latex Output to work

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 20,
    "axes.prop_cycle": plt.cycler('color', ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])})
#plt.style.use('seaborn-poster')
sys.path.append('.')
pd.set_option('display.max_columns', None)


# In[2]:


instance_set = 'all'
instances = None
#instance_set = 'handcrafted'
#instances = [119, 160, 188, 192, 196, 250, 261, 263, 278, 287]


# In[3]:


seeds = [1, 41, 83, 139, 181, 257, 307, 353, 401, 449]


# In[ ]:


file_pattern = f'classic.BARN-{instance_set}/*.classic.*.*/lazy_prm_classic.BARN.instance_*.json'
print(file_pattern)
# Halton results loading
results = collect_deterministic_results([file_pattern], instances, seeds)
classic_table = tabulate_deterministic_results(results)


# In[ ]:


file_pattern = f'bc.BARN-{instance_set}/*.*.*.*/lazy_prm_bc.BARN.instance_*.json'
print(file_pattern)
# Halton results loading
results = collect_deterministic_results([file_pattern], instances, seeds)
bnc_table = tabulate_deterministic_results(results)


# In[ ]:


classic_configs, classic_tables = tabulate_results_by(classic_table, ('check_type', 'step_size', 'direction'))


# In[ ]:


bnc_configs, bnc_tables = tabulate_results_by(bnc_table, ('solver', 'no_good_type', 'direction'))


# In[ ]:


classic_coverage = {}
for v in classic_configs:
    if len(classic_tables[v]) > 0:
        classic_coverage[v] = coverage_over_time(select_valid_instances(classic_tables[v]))


# In[ ]:


relevant_bnc_tables = {v: bnc_tables[v] for v in bnc_configs if v[1] == 'multi_edge'}


# In[ ]:


bnc_coverage = {}
for key, table in relevant_bnc_tables.items():
    bnc_coverage[key] = coverage_over_time(select_valid_instances(table))


# # Coverage comparison

# In[ ]:
plt.rcParams.update({"font.size": 28})

cmp_classic_configs = [('unknown', 0.5, 'bk'), ('unknown', 0.02, 'bk'), ('unknown', 3e-05, 'bk'), ('polytrace', 0.1, 'bk')]
bnc_configs = [('cp_sat', 'multi_edge', 'bk'), ('pulse', 'multi_edge', 'bk')]

f = plt.figure(figsize=(18, 9))
plt.plot([default_time_breakpoints[0],
          default_time_breakpoints[-1]],
         [6000, 6000],
         color='black',
         linewidth=4,
         linestyle='dotted')

for v in cmp_classic_configs:
    label = 'Classic'
    if v[0] == 'polytrace':
        label += ', CHECKSAT, BK'
    else:
        label += ', $\Delta t$={}, BK'.format(v[1])
    plt.plot(default_time_breakpoints, classic_coverage[v], linestyle='-', label=label)

for bnc_config in bnc_configs:
    print(bnc_config)
    label = 'BnC'
    ls = '-.'
    if bnc_config[0] == 'cp_sat':
        label += ', CP-SAT'
    elif bnc_config[0] == 'pulse':
        label += ', PULSE'
        ls = ':'
    if bnc_config[2] == 'bk':
        label += ', BK'
    elif bnc_config[2] == 'gammell':
        label += ', Gammell'
    plt.plot(default_time_breakpoints, bnc_coverage[bnc_config], label=label, linestyle=ls)

#plt.title('Coverage of Lazy PRM configurations')
plt.xlabel('Elapsed Time (s)')
plt.xscale('log')
plt.ylabel('Valid Solutions')
plt.legend()
plt.tight_layout()
plt.show()

f.savefig("classic_vs_bnc_all.pdf", bbox_inches='tight')

# ## Detailed comparison

# In[ ]:


classic_best = classic_tables[('polytrace', 0.1, 'bk')]


# In[ ]:


#bnc_best = bnc_tables[('cp_sat', 'multi_edge', 'gammell')]
bnc_best = bnc_tables[('cp_sat', 'multi_edge', 'bk')]


# In[ ]:


cmp_table = pd.merge(classic_best, bnc_best, on=('instance', 'seed', 'sequence'), suffixes=('.classic', '.bnc'))


# In[ ]:


cmp_table = cmp_table[cmp_table['rgg_V.bnc'] == cmp_table['rgg_V.classic']]


# In[ ]:


cmp_table.head()


# In[ ]:


len(cmp_table)


# ### Plan costs

# In[ ]:


x = cmp_table['smooth_cost_k.classic'].values
y = cmp_table['smooth_cost_k.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Smooth cost comparison', x_name='Classic, POLYTRACE, BK', y_name='BnC, CP-SAT, BK', paper=True)

# %%

different_costs = [v for v in x / y if v != 1.0]
print('Total samples:', len(x), '# Samples with substantive differences:', len(different_costs))

# %%
plt.rcParams.update({
                        "font.size": 20})
f, ax = plt.subplots()

green_diamond = dict(markerfacecolor='g', marker='D')
ax.boxplot([v for v in x / y if v != 1.0],
           vert=False,
            #meanline=True,
            #autorange=True,
            labels=[''],
            flierprops=green_diamond)
ax.set_xlabel(r'$\frac{\Vert \rho_{classic}\Vert}{\Vert \rho_{bnc} \Vert}$')
plt.show()

# %%
f, ax = plt.subplots()

ax.violinplot([v for v in x / y if v != 1.0],
              showmedians=True,
              showextrema=True,
              points=1000)
ax.set_xticklabels([])
ax.set_ylabel(r'${\Vert \rho_{classic}\Vert / \Vert \rho_{bnc} \Vert}$')
plt.show()

f.savefig("classic_vs_bnc_cost_empirical_distro.pdf", bbox_inches='tight')

# %%
print('Median of difference:', np.median(x-y))
print('Std. Dev. of difference:', np.std(x-y))


# In[ ]:


print(lin_model)


# In[ ]:


compare_sequences(x, y)


# ## Plan Length

# In[ ]:


x = cmp_table['plan_length.classic'].values
y = cmp_table['plan_length.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Smooth cost comparison', x_name='Classic', y_name='BnC')


# In[ ]:


print(lin_model)


# ### Graph size

# In[ ]:


x = cmp_table['rgg_E.classic'].values
y = cmp_table['rgg_E.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='RGG Comparison: Edges', x_name='Classic', y_name='BnC')


# In[ ]:


print(lin_model)


# In[ ]:


x = cmp_table['rgg_V.classic'].values
y = cmp_table['rgg_V.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='RGG Comparison: Vertices', x_name='Classic', y_name='BnC')


# In[ ]:


print(lin_model)


# ### Runtime

# In[ ]:


x = cmp_table['elapsed_time.classic'].values
y = cmp_table['elapsed_time.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Runtime comparison', x_name='Classic', y_name='BnC')


# In[ ]:


print(lin_model)


# In[ ]:




