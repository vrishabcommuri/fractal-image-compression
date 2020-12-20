import matplotlib.pyplot as plt


def plot_image(img, title=None, cmap=None, y=0.93):
    plt.figure(figsize=(7,7))
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title, y=y, fontsize=23)
    plt.axis('off')
    plt.show()





# this function did not really work as intended since each chunk subplot
# was given a different scaling by imshow. there is probably a fix for it
# but I didn't take the time to figure it out... leaving it here in
# case it might be useful later.      

# def plot_chunks(chunks, lim=1000):
#     """Plots the chunks returned by chunk_image for sanity checking"""
#     N = chunks.shape[0]
#     if N > lim:
#         N = lim
    
#     fig, ax = plt.subplots(N, N, figsize=(15,15), 
#                            gridspec_kw = {'wspace':0.01, 'hspace':0.01})
    
#     for i in range(N):
#         for j in range(N):
#             plt.suptitle("{}{}".format(i,j))
#             ax[i,j].imshow(chunks[i,j])
#             ax[i,j].axis('off')
            
#     plt.show()