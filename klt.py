from PIL import Image
import numpy as np

def KLT(img_path,block_size,co_num):
    im=Image.open(img_path)
    im=im.convert('L')
    m,n=im.size
 
    im=im.resize(((m//block_size)*block_size, (n//block_size)*block_size))
    pix_mat=np.asarray(im)

    block_num=(m//block_size)*(n//block_size)
    # cal average
    mean=np.zeros(block_size*block_size)
    for x in range(n//block_size):
        for y in range(m//block_size):
          
            mean +=pix_mat[x*block_size:block_size*(x+1),block_size*y:block_size*(y+1)].flatten()
    mean /=block_num
    
    #cal covariance
    C=np.zeros((block_size*block_size,block_size*block_size))
    for x in range(n//block_size):
        for y in range(m//block_size):
            temp_mat=pix_mat[x*block_size:block_size*(x+1),block_size*y:block_size*(y+1)].flatten() -mean
           
            C +=np.matmul(temp_mat.reshape((block_size*block_size,1)),temp_mat.reshape((1,block_size*block_size)))
    C /=block_num
   
   
    eigen_values,eigen_vectors=np.linalg.eig(C)
    
    reconstru_idx=np.zeros(((n//block_size)*co_num,(m//block_size)))
    
    for x in range(n//block_size):
        for y in range(m//block_size):
            temp_mat=pix_mat[x*block_size:block_size*(x+1),block_size*y:block_size*(y+1)].flatten()
            for k in range(co_num):
                reconstru_idx[x*co_num+k][y] = np.sum(temp_mat*eigen_vectors[:,k])


    return eigen_vectors,reconstru_idx

    
if __name__=="__main__":
    block_size=4
    co_num=4
    eigen_vectors,reconstru_idx=KLT('./test.jpg',block_size,co_num)

    img_height=reconstru_idx.shape[1]*block_size
    img_weight=reconstru_idx.shape[0]/co_num*block_size
    img_weight = int(img_weight)
    img_height = int(img_height)
    new_pixmat=np.zeros((img_weight,img_height))
    # reconstruct image
    for x in range(img_weight//block_size):
        for y in range(img_height//block_size):
            reconstru_block=np.zeros(block_size*block_size)
            for k in range(co_num):
                reconstru_block +=reconstru_idx[x*co_num+k][y]*eigen_vectors[:,k]
            new_pixmat[x*block_size:(x+1)*block_size,y*block_size:(y+1)*block_size] = reconstru_block.reshape((block_size,block_size))
    
    
    image=Image.fromarray(new_pixmat.astype('int'))
    image=image.convert('L')
    image.save('out.jpg')
    
