from Adult_utils import *
from fairness_utils import *
import seaborn as sns
from sklearn.metrics import mutual_info_score


data = pd.read_csv('../datasets/adult.csv')
test_data = data.replace('?', np.NaN)
test_data = data_PreProcess(test_data)
#prior to fairness discriminator launch

eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

def MI_heatmap(df, lmbd) :
    target_cols = ['age', 'sex', 'income', 'education.num', 'hours.per.week', 'capital.loss']
    heatmap_data = pd.DataFrame(index=target_cols, columns=target_cols)
    print(df.columns)
    for col1 in target_cols :
        for col2 in target_cols :
            if col1 != 'capital.loss' :
                var_x = np.round(df[col1])
            if col2 != 'capital.loss' :
                var_y = np.round(df[col2])
            phi_coeff = mutual_info_score(var_x, var_y)
            print(col1, ' ', col2, ' : ', phi_coeff)
            heatmap_data.loc[col1, col2] = phi_coeff

    heatmap_data = heatmap_data.apply(pd.to_numeric)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=0, vmax=0.8)
    plt.yticks(rotation=45)
    plt.xticks(rotation=28)
    plt.savefig('heatmaps/heatmap_'+str(lmbd)+'.png')
    plt.show()

MI_heatmap(test_data, 'real')
for i in eps:
    train_data = pd.read_csv('synthetic_data/checkpoint_samples_gan_eps='+str(i)+'/epoch=600.csv')
    MI_heatmap(train_data, i)
