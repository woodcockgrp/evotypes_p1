multi_tree_bic <- function(input.file){

library(PLMIX)

# helper functions
sampleProperly <- function(x, ...) x[sample.int(length(x), ...)]
vecOrder <- function(z, pos){z[z[,pos]==0,pos] <- NA; a <- sort(z[,pos], index.return = TRUE, na.last = TRUE); a$ix[!is.na(a$x)]}

# load in the data
ccf.mat <- input.file 
patient.names <- row.names(ccf.mat)
ccf.mat <- t(ccf.mat)

# remove any patients with zero events
to.keep <- colSums(ccf.mat>0)>0
ccf.mat <- ccf.mat[,to.keep]

# sample trees from raw data
num.runs <- 1000
bic.mat <- matrix(0L, nrow = num.runs, ncol = 10)
all.labels <- matrix(0L, nrow = num.runs, ncol = dim(ccf.mat)[2])

# repeat tree build multiple times
for(j in 1:num.runs){
  print(j)
  
  # assign feature set for this iteration 
  curr.ccf.mat <- ccf.mat
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
        # if we find ourselves in an invalid configuration then throw error
        if(length(new.parent) == 0){
          stop('invalid configuration reached, check CCFs')
          
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
        if(any(is.na(new.order))){stop('position error')}
      }
      
      curr.orders[subclonal.events[new.order],i] <- (num.clonal:(num.clonal+num.subclonal-1))+1
      
    }else if(num.subclonal==1){ # only a single subclonal event
      curr.orders[subclonal.events,i] <- num.clonal+1  
    }
    
  }
  
  # add an event with CCF 1 onto the front to represent carcinogenesis event
  curr.orders[curr.orders>0] <- curr.orders[curr.orders>0]+1
  curr.orders <- rbind(rep(1,dim(curr.orders)[2]), curr.orders) 
  
  #### put data into correct format for PLMIX
  # get a list of events
  event.list <- lapply(1:dim(curr.orders)[2], function(x) vecOrder(curr.orders,x))
  # create a matrix
  event.ranks <- sapply(event.list, `length<-`, max(lengths(event.list)))
  event.ranks[is.na(event.ranks)] <- 0 # change NA to zeros
  event.ranks <- t(event.ranks)
  
  event.ranks <- rank_ord_switch(t(curr.orders), format = "ranking")
  
  # perform BIC
  bic.mat[j,] <- sapply(1:10, function(x) return(tryCatch(mapPLMIX_multistart(event.ranks, K=ncol(event.ranks), G = x, n_start = 2)$mod$bic, error=function(e) NA)))
  # get the class assignments
  pl.model <- invisible(mapPLMIX(event.ranks, K=ncol(event.ranks), G = which(colMeans(bic.mat,na.rm = TRUE)== min(colMeans(bic.mat,na.rm = TRUE))), plot_objective = FALSE))
  
  # fix label inconsistency: largest cardinality corresponds to same class here
  if(sum(pl.model$class_map==1) < sum(pl.model$class_map==2)){
        
        # reverse the labels
        pl.model$class_map[pl.model$class_map==1] <- 0
        pl.model$class_map[pl.model$class_map==2] <- 1
        pl.model$class_map[pl.model$class_map==0] <- 2
    
  }
  # add the class labels into the matrix
  all.labels[j, ] <- pl.model$class_map
  
}

# for each patient, calculate the most frequent class assignments over all the runs
classes <- ifelse(colMeans(all.labels)<1.5,1,2)
names(classes) <- patient.names

browser()
# write the BIC values and class assignemnts to file
write.table(classes, file ='patient_classes.txt', sep='\t', row.names = TRUE, col.names = FALSE, quote = FALSE)
write.table(bic.mat, file = 'bic_mat.txt', sep = '\t', row.names = FALSE, quote = FALSE)

return(classes)
}

