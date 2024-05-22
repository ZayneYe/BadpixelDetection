import matplotlib.pyplot as plt
import seaborn as sns

group_colors = sns.color_palette("deep", 5)

delta = [0.3, 0.4, 0.5, 0.6, 0.7]
recall_dict, precision_dict, iou_dict = {}, {}, {}
for delta_value in delta:
    results_file = 'runs/test/UNet_ISP_{}_fixed/vote_results.txt'.format(delta_value)
    f = open(results_file, 'r')
    recall_dict[delta_value], precision_dict[delta_value], iou_dict[delta_value] = [], [], []
    test_images = []
    for line in f.readlines()[1:-1]:
        n_image, r, p, iou = line.split('\t')
        test_images.append(int(n_image))
        recall_dict[delta_value].append(float(r))
        precision_dict[delta_value].append(float(p))
        iou_dict[delta_value].append(float(iou))

sns.set_style("white")
plt.figure(figsize=(8, 7)) 
# plt.figure(figsize=(20,15))
plt.xlabel('Test Images', size=29)
plt.ylabel('Recall', size=29)
plt.xticks(size=25)
plt.yticks(size=25)
for i, delta_value in enumerate(delta):
    plt.plot(test_images, recall_dict[delta_value], marker='o', markersize=10, linewidth=3.5, color=group_colors[i], label='$\delta$ = {}'.format(delta_value))
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96),
        #   ncol=3, prop ={'size':40})
# plt.legend(loc="best",  prop ={'size':30})
legend=plt.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.54, 0.32), ncol=2)
legend.get_frame().set_alpha(0.5)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=1.5)
plt.savefig('recall.png')


plt.figure(figsize=(8, 7)) 
# plt.figure(figsize=(20,15))
plt.xlabel('Test Images', size=29)
plt.ylabel('Precision', size=29)
plt.xticks(size=25)
plt.yticks(size=25)
for i, delta_value in enumerate(delta):
    plt.plot(test_images, precision_dict[delta_value], marker='o', markersize=10, linewidth=3.5, color=group_colors[i], label='$\delta$ = {}'.format(delta_value))
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96),
        #   ncol=3, prop ={'size':40})
# plt.legend(loc="best",  prop ={'size':30})
legend=plt.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.54, 0.32), ncol=2)
legend.get_frame().set_alpha(0.5)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=1.5)
plt.savefig('precision.png')


# plt.figure(figsize=(20,15))
# plt.xlabel('Test Images', size=65)
# plt.ylabel('Precision', size=65)
# plt.xticks(size=40)
# plt.yticks(size=38)
# for delta_value in delta:
#     plt.plot(test_images, precision_dict[delta_value], marker='o', markersize=25, linewidth=5, label='$\delta$ = {}'.format(delta_value))
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96),
#           ncol=3, prop ={'size':40})
# plt.tight_layout()
# plt.grid()
# plt.savefig('precision.png')

plt.figure(figsize=(8,6))
plt.xlabel('Test Images', size=16)
plt.ylabel('IoU', size=16)
plt.xticks(size=16)
plt.yticks(size=16)
for delta_value in delta:
    plt.plot(test_images, iou_dict[delta_value], "-o", label='delta = {}'.format(delta_value))
plt.legend(loc="lower right",  prop ={'size':15})
plt.grid()
plt.savefig('iou.png')