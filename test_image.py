import plt
def viz_textbb(name,text_im, charBB_list, wordBB,senBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1,figsize=(15,15))
    plt.imshow(text_im)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    # for i in range(len(charBB_list)):
    #     bbs = charBB_list[i]
    #     ni = bbs.shape[-1]
    #     for j in range(ni):
    #         bb = bbs[:,:,j]
    #         bb = np.c_[bb,bb[:,0]]
    #         plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # # plot the word-BB:
    # for i in range(wordBB.shape[-1]):
    #     bb = wordBB[:,:,i]
    #     bb = np.c_[bb,bb[:,0]]
    #     plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
    #     # visualize the indiv vertices:
    #     vcol = ['r','g','b','k']
    #     for j in range(4):
    #         plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

    for i in range(senBB.shape[-1]):
        bb = senBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    # plt.show(block=False)
    plt.savefig('img/' + name + '.jpg')