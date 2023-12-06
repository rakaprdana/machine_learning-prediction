import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Baca dataset (gantilah 'nama_file.csv' dengan nama file dataset Anda)
df = pd.read_csv('./data/sample2.csv', parse_dates=['DateTime'])

# Contoh visualisasi line chart time series
plt.figure(figsize=(10, 6))
sns.lineplot(x='DateTime', y='Vehicles', data=df)
plt.title('Line Chart Time Series')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Kendaraan')

plt.xlim(pd.Timestamp('2015-11-01'), pd.Timestamp('2015-11-30'))
plt.grid(True)
plt.show()
