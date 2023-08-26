multi_sample_pl <- function(input.file, patient.labels = NULL){
      
library(PlackettLuce)
      
sampleProperly <- function(x, ...) x[sample.int(length(x), ...)]
vecOrder <- function(z, pos){z[z[,pos]==0,pos] <- NA; a <- sort(z[,pos], index.return = TRUE, na.last = TRUE); a$ix[!is.na(a$x)]}

# flag to impute the data missing due to censorship for each class
impute.additional.data <- FALSE

# load in the patient labels if not input
if(is.null(patient.labels)){
      patient.labels <- read.table('./patient_classes.txt', sep='\t', stringsAsFactors = FALSE, row.names = 1, header = FALSE)
      patient.labels <- setNames(patient.labels[,1], rownames(patient.labels))
}


#feature.names <-  names(ccf.mat)[1:26]
ccf.mat <- t(input.file)

# remove any patients with single or zero events
to.keep <- colSums(ccf.mat>0)>1

ccf.mat <- ccf.mat[,to.keep]
patient.labels <- patient.labels[to.keep]

num.runs <- 1000

for(clust in unique(patient.labels)){
      
      coefs <- matrix(0L, nrow = num.runs, ncol = dim(ccf.mat)[1])  
      
      cluster.ccf.mat <- ccf.mat[,which(patient.labels==clust)]
      
      # sample a ranking for all cloncal and subclonal events separately
      for(j in 1:num.runs){
            print(j)
            
            # assign feature set for this iteration (as we might change it)
            curr.ccf.mat <- cluster.ccf.mat

            # initialise ranking matrix for Plackett Luce input 
            curr.orders <- curr.ccf.mat
            
            ## for each patient, create an order of events that is consistent with the CCF 
            ## values by working through each event in CCF order, finding a suitable place
            ## for the event on the tree and sampling an rank consistent with this
            for(i in 1:dim(curr.ccf.mat)[2]){   
                  
                  # identify clonal events and order them randomly
                  clonal.events <- which(curr.ccf.mat[,i]==1)
                  num.clonal <- length(clonal.events)
                  if(num.clonal>1){
                        curr.orders[clonal.events,i] <- sample((1:num.clonal))
                  }
                  
                  # identify subclonal events
                  subclonal.events <- which(curr.ccf.mat[,i]<1 & curr.ccf.mat[,i]>0)
                  num.subclonal <- length(subclonal.events)
                  
                  if(num.subclonal>1){
                        
                        # order the events by CCF
                        orig.order <- order(curr.ccf.mat[subclonal.events,i], decreasing = TRUE)
                        subclonal.events.sorted <- subclonal.events[orig.order]
                        subclonal.ccf <- curr.ccf.mat[subclonal.events.sorted,i]
                        
                        # create vectors to store the necessary information
                        new.order <- orig.order 
                        max.child <- rep(num.subclonal,num.subclonal)  
                        
                        avail.parent <- seq_len(num.subclonal+1)
                        
                        # calculate the possible parental CCFs (from a clonal root)
                        parental.ccf <- c(1,subclonal.ccf)
                        # for each potential child, sample a parent
                        for(new.child in seq_len(num.subclonal)){
                              
                              #### sample parent (condition: child CCF < available CCF of parent, 
                              ####                  order can never be greater than it is now)
                              
                              if(new.child >1){
                                    # if same CCF then same event so sample from same events as parent
                                    if(subclonal.ccf[new.child]==subclonal.ccf[new.child-1]){
                                          new.parent <- sampleProperly(which(parental.ccf[1:new.child]==subclonal.ccf[new.child]),1)
                                          same.event <- TRUE
                                    }else{ # sample parent
                                          
                                          new.parent <- sampleProperly(which(parental.ccf>subclonal.ccf[new.child]),1)
                                          same.event <- FALSE
                                          
                                    }
                              }else{ # first sample so parent is root node
                                    new.parent <- 1
                                    same.event <- FALSE
                              }
                              # if we find ourselves in an invalid configuration then restart
                              if(length(new.parent) == 0){
                                    print("error")
                                    browser()
                                    
                              }
                              
                              # update the available parent CCF
                              parental.ccf[new.parent] <- parental.ccf[new.parent] - subclonal.ccf[new.child]
                              
                              #### sample an order position (condition: after parent)
                              
                              # identify where the parent is ordered currently
                              parent.pos <- which(new.order==orig.order[new.parent-1])
                              
                              # deal with special situations
                              if(same.event){ # happens either immediately before or after parent
                                    
                                    new.child.pos <- sampleProperly(parent.pos:min((parent.pos+1),num.subclonal),1)
                                    
                                    # if before then give the parent the CCF
                                    if(new.child.pos==parent.pos){
                                          parental.ccf <- replace(parental.ccf, c(new.child+1, new.parent), parental.ccf[c(new.parent, new.child+1)])
                                    }
                              }else if(length(parent.pos)==0){ # root node (sample from all)
                                    
                                    new.child.pos <- sampleProperly(1:(num.subclonal),1) 
                                    
                              }else if(parent.pos == num.subclonal){ # parent is last
                                    
                                    new.child.pos <- num.subclonal
                                    
                              }else{ # everything else
                                    
                                    new.child.pos <- sampleProperly((parent.pos+1):(num.subclonal),1)
                                    
                              }
                              
                              # remove this sample from the order vector
                              new.order <- new.order[-which(new.order==orig.order[new.child])]
                              # and re-add it in the new position
                              if(new.child.pos == 1){
                                    new.order <- c(orig.order[new.child], new.order)
                              }else{
                                    new.order <- c(new.order[1:(new.child.pos-1)], orig.order[new.child], new.order[-c(1:(new.child.pos-1))])
                              }
                              if(any(is.na(new.order))){browser()} # break and work out what is going on
                              
                        }
                        
                        curr.orders[subclonal.events[new.order],i] <- (num.clonal:(num.clonal+num.subclonal-1))+1
                        
                  }else if(num.subclonal==1){ # only a single subclonal event
                        curr.orders[subclonal.events,i] <- num.clonal+1  
                  }
                  
                  #### missing data ####
                  # augment data with additional events sampled with proportion to their frequency in their subgroups
                  
                  if(impute.additional.data){
                        ## remove current patient from the imputation
                        
                        # calculate the frequencies of each event for this subset
                        feature.freqs <- rowSums(curr.ccf.mat[,-i]>0)
                        clonal.feature.freqs <- rowSums(curr.ccf.mat[,-i]==1)
                        subclonal.feature.freqs <- rowSums(curr.ccf.mat[,-i]<1 & curr.ccf.mat[,-i] >0)
                        
                        # probabilities are the frequencies of subclonal occurrences over the total frequencies
                        sampling.prob <- (subclonal.feature.freqs/feature.freqs)#*(feature.freqs/num.patients)
                        sampling.prob[!is.finite(sampling.prob)] <- 0
                        additional.events <- which(runif(length(feature.freqs))<sampling.prob)#feature.freqs/num.patients)
                        # cull those already present
                        additional.events <- setdiff(additional.events, intersect(additional.events, which(curr.ccf.mat[,i]>0)))
                        num.additional <- length(additional.events)
                        # add them at the end in a random order
                        if(num.additional>1){
                              curr.orders[additional.events,i] <- sample(((max(curr.orders[,i])):(max(curr.orders[,i])+num.additional-1))+1)
                        }else if(num.subclonal==1){ # only a single additional event
                              curr.orders[additional.events,i] <- max(curr.orders[,i])+1
                        }
                  }
                  
            }
            
            # add an event with CCF 1 onto the front to represent carcinogenesis event
            curr.orders[curr.orders>0] <- curr.orders[curr.orders>0]+1
            curr.orders <- rbind(rep(1,dim(curr.orders)[2]), curr.orders)
            
            # calculate the PL coefficients
            pl.model <- PlackettLuce(t(curr.orders), npseudo = 0.01, maxit = 1000);
            coefs[j,] <- coef(pl.model)[-1]
            curr.orders <- curr.orders[-1,]      
            
      }
      
      # pack the proportion information into the same variable
      feature.props <-rowMeans(cluster.ccf.mat>0)
      coefs <- rbind(coefs, feature.props)
      coefs <- as.data.frame(coefs)

      write.table(coefs, file = paste0('pl_ordering_', clust ,'.txt'), sep = '\t', row.names = FALSE, quote = FALSE)
      
      
} # end clust loop!

}
# debugging plot code
#qv <- qvcalc(mod)
#plot(qv, xlab = "Aberration", ylab = "Worth (log)", main = NULL)
