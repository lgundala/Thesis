import random
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import io
file = pd.read_excel('/Users/Amulya/Documents/Depp_results_roc.xlsx', sheet='Sheet1');
one_ten = file.loc[file['size']=="1/10"];
one_hundred = file.loc[file['size']=="1/100"];
one_thousand = file.loc[file['size']=="1/1000"];

df = file
print(df);
# Setting the positions and width for the bars
pos = list(range(len(df['Balanced'])))
print(pos);
width = 0.1

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        #using df['pre_score'] data,
        df['Balanced'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#EE3224',
        # with label the first value in first_name
        label=df['size'][0])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df['B1+B2+B3'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the second value in first_name
        label=df['size'][1])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df['B1'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#808000',
        # with label the third value in first_name
        label=df['size'][2])
plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df['B1'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#008000',
        # with label the third value in first_name
        label=df['size'][3])
plt.bar([p + width*4 for p in pos],
        #using df['post_score'] data,
        df['B3'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#080808',
        # with label the third value in first_name
        label=df['size'][3])

# Set the y axis label
ax.set_ylabel('roc_auc_score')

# Set the chart's title
ax.set_title('Comparison of roc scores')

# Set the position of the x ticks
ax.set_xticks([p + 1 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['size'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*6)
plt.ylim([0, 1])

# Adding the legend and showing the plot
plt.legend(['Balanced','B1+B2+B3','B1','B2','B3'], loc='upper left')
plt.grid()
plt.show()

