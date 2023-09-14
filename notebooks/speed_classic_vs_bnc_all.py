#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
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


# In[4]:


file_pattern = f'classic.speed.BARN-{instance_set}/*.classic.*.*/lazy_prm_classic.BARN.instance_*.json'
print(file_pattern)
# Halton results loading
results = collect_deterministic_results([file_pattern], instances, seeds)
classic_table = tabulate_deterministic_results(results)


# In[5]:


file_pattern = f'bc.speed.BARN-{instance_set}/*.*.*.*/lazy_prm_bc.BARN.instance_*.json'
print(file_pattern)
# Halton results loading
results = collect_deterministic_results([file_pattern], instances, seeds)
bnc_table = tabulate_deterministic_results(results)


# In[6]:


classic_configs, classic_tables = tabulate_results_by(classic_table, ('check_type','step_size','direction', 'max_speed'))


# In[7]:


bnc_configs, bnc_tables = tabulate_results_by(bnc_table, ('solver','no_good_type','direction', 'max_speed'))


# In[8]:


classic_coverage = {}
for v in classic_configs:
    if len(classic_tables[v]) > 0:
        classic_coverage[v] = coverage_over_time(select_valid_instances(classic_tables[v]))


# In[9]:


relevant_bnc_tables = {v: bnc_tables[v] for v in bnc_configs if v[1] == 'multi_edge'}


# In[10]:


bnc_coverage = {}
for key, table in relevant_bnc_tables.items():
    bnc_coverage[key] = coverage_over_time(select_valid_instances(table))


# # Coverage comparison, $v_{max} \leq 1.5$

# In[11]:


cmp_classic_configs = [('unknown', 0.02, 'bk', 1.5), ('polytrace', 0.1, 'bk', 1.5)]
bnc_configs = [('cp_sat', 'multi_edge', 'bk', 1.5), ('pulse', 'multi_edge', 'bk', 1.5)]

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
        label += ', CHECKSAT'
    else:
        label += ', $\Delta t$={}'.format(v[1])
    label += ', ${} = {}$'.format('v_{max}', v[-1])
    plt.plot(default_time_breakpoints, classic_coverage[v], label=label)

for bnc_config in bnc_configs:
    print(bnc_config)
    label = 'BnC'
    if bnc_config[0] == 'cp_sat':
        label += ', CP-SAT'
    elif bnc_config[0] == 'pulse':
        label += ', PULSE'
    if bnc_config[2] == 'bk':
        label += ', BK'
    elif bnc_config[2] == 'gammell':
        label += ', Gammell'
    label += ', ${} = {}$'.format('v_{max}', bnc_config[-1])
    plt.plot(default_time_breakpoints, bnc_coverage[bnc_config], label=label)

#plt.title('Coverage of Lazy PRM configurations')
plt.xlabel('Elapsed Time (s)')
plt.xscale('log')
plt.ylabel('Valid Solutions')
plt.legend()
plt.tight_layout()
plt.show()

f.savefig("classic_vs_bnc_all_speed_1_5.pdf", bbox_inches='tight')

# ## Detailed comparison, $v_{max} \leq 1.5$

# In[12]:


classic_best = classic_tables[('polytrace', 0.1, 'bk', 1.5)]
bnc_best = bnc_tables[('pulse', 'multi_edge', 'bk', 1.5)]
cmp_table = pd.merge(classic_best, bnc_best, on=('instance', 'seed', 'sequence'), suffixes=('.classic', '.bnc'))
cmp_table.head()


# ### Plan costs

# In[13]:


x = cmp_table['smooth_cost_k.classic'].values
y = cmp_table['smooth_cost_k.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Smooth cost comparison', x_name='Classic', y_name='BnC')
print(lin_model)


# In[14]:


print(lin_model)


# ## Plan Length

# In[15]:


x = cmp_table['plan_length.classic'].values
y = cmp_table['plan_length.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Smooth cost comparison', x_name='Classic', y_name='BnC')


# In[16]:


print(lin_model)


# ### Graph size

# In[17]:


x = cmp_table['rgg_E.classic'].values
y = cmp_table['rgg_E.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='RGG Comparison: Edges', x_name='Classic', y_name='BnC')

# %%

different_costs = [v for v in x / y if v != 1.0]
print('Total samples:', len(x), '# Samples with substantive differences:', len(different_costs))

# %%
f, ax = plt.subplots()

ax.violinplot([v for v in x / y if v != 1.0],
              showmedians=True,
              showextrema=True,
              points=1000)
ax.set_xticklabels([])
ax.set_ylabel(r'${\Vert \rho_{classic}\Vert / \Vert \rho_{bnc} \Vert}$')
plt.show()

f.savefig("classic_vs_bnc_cost_empirical_distro_1_5.pdf", bbox_inches='tight')

# In[18]:


print(lin_model)


# In[19]:


x = cmp_table['rgg_V.classic'].values
y = cmp_table['rgg_V.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='RGG Comparison: Vertices', x_name='Classic', y_name='BnC')


# In[20]:


print(lin_model)


# ### Runtime

# In[21]:


x = cmp_table['elapsed_time.classic'].values
y = cmp_table['elapsed_time.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Runtime comparison', x_name='Classic', y_name='BnC')


# In[22]:


print(lin_model)


# # Coverage Comparison, $v_{max} \leq 2$

# In[23]:


cmp_classic_configs = [('unknown', 0.02, 'bk', 2.0), ('polytrace', 0.1, 'bk', 2.0)]
bnc_configs = [('cp_sat', 'multi_edge', 'bk', 2.0), ('pulse', 'multi_edge', 'bk', 2.0)]

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
        label += ', CHECKSAT'
    else:
        label += ', $dt$={}'.format(v[1])
    label += ', ${} = {}$'.format('v_{max}', v[-1])
    plt.plot(default_time_breakpoints, classic_coverage[v], label=label)

for bnc_config in bnc_configs:
    print(bnc_config)
    label = 'BnC'
    if bnc_config[0] == 'cp_sat':
        label += ', CP-SAT'
    elif bnc_config[0] == 'pulse':
        label += ', PULSE'
    if bnc_config[2] == 'bk':
        label += ', BK'
    elif bnc_config[2] == 'gammell':
        label += ', Gammell'
    label += ', ${} = {}$'.format('v_{max}', bnc_config[-1])
    plt.plot(default_time_breakpoints, bnc_coverage[bnc_config], label=label)

#plt.title('Coverage of Lazy PRM configurations')
plt.xlabel('Elapsed Time (s)')
plt.xscale('log')
plt.ylabel('Valid Solutions')
plt.legend()
plt.tight_layout()
plt.show()

f.savefig("classic_vs_bnc_speed_2_0.pdf", bbox_inches='tight')

# In[24]:


## Detailed comparison, $v_{max} \leq 1.5$


# In[25]:


classic_best = classic_tables[('polytrace', 0.1, 'bk', 2.0)]
bnc_best = bnc_tables[('pulse', 'multi_edge', 'bk', 2.0)]
cmp_table = pd.merge(classic_best, bnc_best, on=('instance', 'seed', 'sequence'), suffixes=('.classic', '.bnc'))
cmp_table.head()


# ## Plan costs

# In[26]:


x = cmp_table['smooth_cost_k.classic'].values
y = cmp_table['smooth_cost_k.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Smooth cost comparison', x_name='Classic', y_name='BnC')
print(lin_model)

f, ax = plt.subplots()

ax.violinplot([v for v in x / y if v != 1.0],
              showmedians=True,
              showextrema=True,
              points=1000)
ax.set_xticklabels([])
ax.set_ylabel(r'${\Vert \rho_{classic}\Vert / \Vert \rho_{bnc} \Vert}$')
plt.show()

f.savefig("classic_vs_bnc_cost_empirical_distro_2_0.pdf", bbox_inches='tight')


# ## Plan Length

# In[27]:


x = cmp_table['plan_length.classic'].values
y = cmp_table['plan_length.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Smooth cost comparison', x_name='Classic', y_name='BnC')


# ## Graph Size

# In[28]:


x = cmp_table['rgg_E.classic'].values
y = cmp_table['rgg_E.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='RGG Comparison: Edges', x_name='Classic', y_name='BnC')
print(lin_model)


# In[29]:


x = cmp_table['rgg_V.classic'].values
y = cmp_table['rgg_V.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='RGG Comparison: Vertices', x_name='Classic', y_name='BnC')
print(lin_model)


# ## Runtime

# In[30]:


x = cmp_table['elapsed_time.classic'].values
y = cmp_table['elapsed_time.bnc'].values

lin_model = compare_with_scatter_chart(x, y, title='Runtime comparison', x_name='Classic', y_name='BnC')

