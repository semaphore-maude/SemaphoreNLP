# plot libraries
import plotly.offline as py
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from collections import OrderedDict

class PlotWordCount(object):
    def __init__(self, count_dict):
        # order by value to plot
        self.count_dict = count_dict
        
    def order_count(self):   
        count_dict = self.count_dict
        freqs = count_dict.items()
        sorted_count = sorted(freqs, key=lambda t: t[1], reverse=True)
        return sorted_count
    
    def freq_plot(self, top_n=50,
                        width=1.0,
                        c_scale='Portland',
                        title = 'Top word frequencies (after cleanup and lemmatization)',
                        save_filename='word_count_bar',
                        image_format ='png'):

        ordered_count = self.order_count()
        sorted_word = [d[0] for d in ordered_count[:top_n]]
        sorted_freq = [d[1] for d in ordered_count[:top_n]]

        data_word = [go.Bar(x = sorted_word,
                            y = sorted_freq,
                            marker = dict(colorscale=c_scale,
                                          color=sorted_freq,
                                          line=dict(color='rgb(0,0,0)', width=width)),
                    text='Word count')]
        
        layout = go.Layout(title=title)        
        fig = go.Figure(data=data_word, layout=layout)
        py.iplot(fig, filename=save_filename, image=image_format);


    def cloud_plot(self, size=(9,6),
                         background_color="black", 
                         max_words=1000, 
                         stopwords=[], 
                         max_font_size= 60,
                         min_font_size=5,
                         collocations = False,
                         colormap="coolwarm",
                         plot_title="Most common words",
                         plot_fontsize=30,
                         interpolation='bilinear',
                         save_filename='results/visualization/cloudplot.png'):

        self.text_cloud = " ".join(word for word in self.count_dict.elements())

        plt.figure(figsize=size);
        wc = WordCloud(background_color=background_color, 
                       max_words=max_words, 
                       stopwords=stopwords, 
                       max_font_size=max_font_size,
                       min_font_size=min_font_size,
                       collocations = collocations,
                       colormap=colormap);
        
        wc.generate(self.text_cloud);
        # store to file
        wc.to_file(save_filename);
        plt.title(plot_title, fontsize=plot_fontsize);
        plt.imshow(wc, interpolation=interpolation);

        plt.margins(x=0.25, y=0.25);
        plt.axis('off');
        plt.show();
        plt.savefig(save_filename);




        
