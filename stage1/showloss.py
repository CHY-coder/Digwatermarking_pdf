import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(log_file_path):
    # Initialize data storage
    data = {
        'epoch': [],
        'batch': [],
        'total_loss': [],
        'visual_quality': [],
        'perceptual': [],
        'discriminator': [],
        'message': []
    }

    # Compile a pattern to extract data
    pattern = re.compile(r'Epoch (\d+):.*\[(\d+)/\d+\]\ttotal loss: ([\d.]+)\tvisual quality: ([\d.]+)\tperceptual: ([\d.e-]+)\tdiscriminator: ([\d.]+)\tmessage: ([\d.]+)')

    # Read and process the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                data['epoch'].append(int(match.group(1)))
                data['batch'].append(int(match.group(2)))
                data['total_loss'].append(float(match.group(3)))
                data['visual_quality'].append(float(match.group(4)))
                data['perceptual'].append(float(match.group(5)))
                data['discriminator'].append(float(match.group(6)))
                data['message'].append(float(match.group(7)))

    df = pd.DataFrame(data)
    epoch_means = df.groupby('epoch').mean().reset_index()

    # Plotting
    plt.figure(figsize=(14, 8))
    colors = ['blue', 'green', 'red', 'orange', 'cyan']

    for i, column in enumerate(['total_loss', 'visual_quality', 'perceptual', 'discriminator', 'message']):
        plt.plot(epoch_means['epoch'], epoch_means[column], label=column, color=colors[i], linewidth=2)
        plt.text(epoch_means['epoch'].iloc[-1], epoch_means[column].iloc[-1], f'{column}', fontsize=9, color=colors[i], ha='right', va='center')

    plt.title('Losses per Epoch with Inline Labels and Updated Colors')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file_path>")
    else:
        log_file_path = sys.argv[1]
        plot_losses(log_file_path)
