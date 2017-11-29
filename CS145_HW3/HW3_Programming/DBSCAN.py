# =======================================================================
from KMeans import KMeans
from DataPoints import DataPoints
import random
import math

import matplotlib.pyplot as plt
# =======================================================================
class DBSCAN:
    # -------------------------------------------------------------------
    def __init__(self):
        self.e = 0.0
        self.minPts = 3
        self.noOfLabels = 0
        self.fp = ""
    # -------------------------------------------------------------------
    def main(self, args):
        seed = 71
        print("For dataset1")
        dataSet = KMeans.readDataSet("dataset1.txt")
        random.Random(seed).shuffle(dataSet)
        self.noOfLabels = DataPoints.getNoOFLabels(dataSet)
        self.e = self.getEpsilon(dataSet)
        print("Esp :" + str(self.e))
        self.fp = "DBSCAN_dataset.csv"
        self.dbscan(dataSet)

        print("\nFor dataset2")
        dataSet = KMeans.readDataSet("dataset2.txt")
        random.Random(seed).shuffle(dataSet)
        self.noOfLabels = DataPoints.getNoOFLabels(dataSet)
        self.e = self.getEpsilon(dataSet)
        print("Esp :" + str(self.e))
        self.fp = "DBSCAN_dataset2.csv"
        self.dbscan(dataSet)

        print("\nFor dataset3")
        dataSet = KMeans.readDataSet("dataset3.txt")
        random.Random(seed).shuffle(dataSet)
        self.noOfLabels = DataPoints.getNoOFLabels(dataSet)
        self.e = self.getEpsilon(dataSet)
        print("Esp :" + str(self.e))
        self.fp = "DBSCAN_dataset3.csv"
        self.dbscan(dataSet)
    # -------------------------------------------------------------------
    def getEpsilon(self, dataSet):
        sumOfDist = 0.0
        # ****************Please Fill Missing Lines Here*****************
        dist = [0.0 for x in range(len(dataSet))]
        for point in dataSet:
            for i in range(len(dataSet)):
                dist[i] = math.sqrt((point.x - dataSet[i].x)**2+(point.y - dataSet[i].y)**2)
            dist.sort()
            # print dist
            for j in range(1,2*self.minPts+1):
                sumOfDist = sumOfDist + dist[j]/self.minPts
        return 1.05*sumOfDist/len(dataSet)
    # -------------------------------------------------------------------
    def dbscan(self, dataSet):
        clusters = []
        visited = set()
        noise = set()

        # Iterate over data points
        for i in range(len(dataSet)):
            point = dataSet[i]
            if point in visited:
                continue
            visited.add(point)
            N = []
            minPtsNeighbours = 0

            # check which point satisfies minPts condition 
            for j in range(len(dataSet)):
                if i==j:
                    continue
                pt = dataSet[j]
                dist = self.getEuclideanDist(point.x, point.y, pt.x, pt.y)
                if dist <= self.e:
                    minPtsNeighbours += 1
                    N.append(pt)

            if minPtsNeighbours >= self.minPts:
                cluster = set()
                cluster.add(point)
                point.isAssignedToCluster = True

                j = 0
                while j < len(N):
                    point1 = N[j]
                    minPtsNeighbours1 = 0
                    N1 = []
                    if not point1 in visited:
                        visited.add(point1)
                        for l in range(len(dataSet)):
                            pt = dataSet[l]
                            dist = self.getEuclideanDist(point1.x, point1.y, pt.x, pt.y)
                            if dist <= self.e:
                                minPtsNeighbours1 += 1
                                N1.append(pt)
                        if minPtsNeighbours1 >= self.minPts:
                            self.removeDuplicates(N, N1)
                        else:
                            N1 = []
                    # Add point1 is not yet member of any other cluster then add it to cluster
                    if not point1.isAssignedToCluster:
                        cluster.add(point1)
                        point1.isAssignedToCluster = True
                    j += 1
                # add cluster to the list of clusters
                clusters.append(cluster)

            else:
                noise.add(point)

            N = []

        # List clusters
        print("Number of clusters formed :" + str(len(clusters)))
        print("Noise points :" + str(len(noise)))

        # Calculate purity
        maxLabelCluster = []
        for j in range(len(clusters)):
            maxLabelCluster.append(KMeans.getMaxClusterLabel(clusters[j]))
        purity = 0.0
        for j in range(len(clusters)):
            purity += maxLabelCluster[j]
        purity /= len(dataSet)
        print("Purity is :" + str(purity))

        nmiMatrix = DataPoints.getNMIMatrix(clusters, self.noOfLabels)
        nmi = DataPoints.calcNMI(nmiMatrix)
        print("NMI :" + str(nmi))

        color_idx = 0
        colors = ['b','g','r','c','m','y','k','w']  
        minX=100000
        minY=100000
        maxX=0
        maxY=0 
        for cluster in clusters:        
            for point in cluster: 
                if(point.x<=minX):
                    minX=point.x 
                if(point.y<=minY):
                    minY=point.y 
                if(point.x>maxX):
                    maxX=point.x 
                if(point.y>maxY):
                    maxY=point.y
                plt.scatter(point.x, point.y, c=colors[color_idx%8])
            color_idx += 1
        plt.axis([minX-1, maxX+1, minY-1, maxY+1])
        # plt.show()
        figname = self.fp[:len(self.fp)-4] + ".png"
        plt.savefig(figname)
        plt.gcf().clear()

        DataPoints.writeToFile(noise, clusters, "DBSCAN_dataset3.csv")
    # -------------------------------------------------------------------
    def removeDuplicates(self, n, n1):
        for point in n1:
            isDup = False
            for point1 in n:
                if point1 == point:
                    isDup = True
            if not isDup:
                n.append(point)
    # -------------------------------------------------------------------
    def getEuclideanDist(self, x1, y1, x2, y2):
        dist = math.sqrt(pow((x2-x1), 2) + pow((y2-y1), 2))
        return dist
# =======================================================================
if __name__ == "__main__":
    d = DBSCAN()
    d.main(None)