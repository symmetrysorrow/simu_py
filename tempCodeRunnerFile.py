    plt.xlabel("Current[A]", fontsize=15)
        plt.ylabel("Count", fontsize=15)
        plt.tight_layout()
        plt.legend(labelspacing=0, fontsize=8, markerscale=0.5)
        plt.savefig(f"{Data_path}/energy_st_histgram_{target}_{para['E']}.png")
        if show:
            plt.show()
            plt.plot(para["position"], ene_reso_st)
            plt.show()
        else:
            plt.clf()