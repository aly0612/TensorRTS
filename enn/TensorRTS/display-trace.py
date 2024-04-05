import ast
from colorama import Fore, Style

class PrintStuff:
    def __init__(self):
        self.mapsize: int = 32
        self.clusters: list[list[int]] = []
        self.tensors:  list[list[int]] = []

    def print_universe(self):
            print(Fore.BLUE, end="")
            for j in range(self.mapsize*2) :
                print("─", end="")
            print(Fore.RESET + "─" + Fore.RED, end="")
            for j in range(self.mapsize*2) :
                print("─", end="")
            print(Fore.RESET)


            # Board indexes
            for j in range(self.mapsize):
                print(f" {j:>2} ", end="")
            print("")

            # Top
            print("┌─",end="")
            for j in range(self.mapsize-1):
                print(f"──┬─", end="")
            print("──┐")

            # Middle
            print("│ ",end="")
            cluster_idx = 0
            cluster_size = len(self.clusters)
            # forgive me for this
            for i in range(self.mapsize):
                if i == self.tensors[0][0]:
                    print(Fore.BLUE + f"●" + Fore.RESET,end="")
                    if (i == self.clusters[cluster_idx][0]): cluster_idx += 1
                elif i == self.tensors[1][0]:
                    print(Fore.RED + f"●" + Fore.RESET,end="")
                    if (i == self.clusters[cluster_idx][0]): cluster_idx += 1
                elif cluster_idx < cluster_size and i == self.clusters[cluster_idx][0]:
                    print(f"{self.clusters[cluster_idx][1]}",end="")
                    if cluster_idx < len(self.clusters)-1:
                        cluster_idx += 1
                else:
                    print(" ",end="")
                print(" │ ",end="")
            print("")

            # Bottom
            print("└─",end="")
            for j in range(self.mapsize-1):
                print(f"──┴─", end="")
            print("──┘\n")

            print("Pos\tDim\tx\ty\tMP")

            print(Fore.BLUE + Style.BRIGHT,end="")
            for i in range(len(self.tensors[0])):
                print(f"{self.tensors[0][i]}\t", end="")
            print(f"{self.tensors[0][3]*self.tensors[0][3]+self.tensors[0][2]}\t", end="")
            print(Fore.RESET+Style.RESET_ALL)

            print(Fore.RED + Style.BRIGHT,end="")
            for i in range(len(self.tensors[1])):
                print(f"{self.tensors[1][i]}\t", end="")
            print(f"{self.tensors[1][2]*self.tensors[1][2]+self.tensors[1][3]}\t", end="")
            print(Fore.RESET+Style.RESET_ALL+"\n\n")

    def main(self):
        state = 0
        with open("GameTrace.txt", "r") as file:
            cluster_line = file.readline()
            tensor_line = file.readline()
            while cluster_line and tensor_line:
                print("Gamestate " + Style.BRIGHT + str(state) + Style.RESET_ALL + '\n')
                self.clusters = ast.literal_eval(cluster_line)
                self.tensors = ast.literal_eval(tensor_line)
                self.print_universe()
                cluster_line = file.readline()
                tensor_line = file.readline()
                state += 1


if __name__ == "__main__":
    x = PrintStuff()
    x.main()
    