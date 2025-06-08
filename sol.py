import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Step 3: Create the Venn diagram
plt.figure(figsize=(8, 6))

# Define labels
venn = venn3(
    subsets=(1, 1, 1, 0, 0, 0, 1),  # Only 3 individual sets and full intersection are non-zero
    set_labels=('AI Accuracy\n(SlowFast, I3D)', 'Real-Time Detection', 'Alert System')
)

# Customize intersection labels
venn.get_label_by_id('100').set_text('SlowFast / I3D only')
venn.get_label_by_id('010').set_text('Real-Time only')
venn.get_label_by_id('001').set_text('Alert only')
venn.get_label_by_id('111').set_text('Your System')

# Hide unused subset labels
for subset_id in ['110', '101', '011']:
    label = venn.get_label_by_id(subset_id)
    if label:
        label.set_text('')

# Add title
plt.title("Venn Diagram of Your AI System")

# Show the plot
plt.show()