import os
import matplotlib.pyplot as plt

def plot_trajectory(trajectory_data, plt_save_path, step):
    prices = trajectory_data[:, :5]
    plt.figure()
    plt.subplot(511)
    plt.plot(prices[:, 0])                
    plt.subplot(512)
    plt.plot(prices[:, 1])
    plt.subplot(513)
    plt.plot(prices[:, 2])
    plt.subplot(514)
    plt.plot(prices[:, 3])
    plt.subplot(515)
    plt.plot(prices[:, 4])
    plt.savefig(os.path.join(plt_save_path, f'prices_{step}.png'))

    accounts = trajectory_data[:, 5:]
    plt.figure()
    plt.subplot(711)
    plt.plot(accounts[:, 6])                
    plt.subplot(712)
    plt.plot(accounts[:, 0])
    plt.subplot(713)
    plt.plot(accounts[:, 1])
    plt.subplot(714)
    plt.plot(accounts[:, 2])
    plt.subplot(715)
    plt.plot(accounts[:, 3])
    plt.subplot(716)
    plt.plot(accounts[:, 4])
    plt.subplot(717)
    plt.plot(accounts[:, 5])             
    plt.savefig(os.path.join(plt_save_path, f'account_{step}.png'))

    plt.close('all')