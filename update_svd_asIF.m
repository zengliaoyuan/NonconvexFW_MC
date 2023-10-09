function Xnew = update_svd_asIF(Xold, u_add, v_add)
%  update the svd decomposition for rank 1 updation matrix
%  The lines in this function are a part of the "update_svd.m" file in the InFaceExtended-MatrixCompletion solver


Uold = Xold.U;
dold = Xold.d;
Vold = Xold.V;

if isempty(Uold) && isempty(Vold) && isempty(dold)
    u_norm = norm(u_add, 2);
    v_norm = norm(v_add, 2);
    Unew = u_add/u_norm;
    Vnew = v_add/v_norm;
    Dnew = u_norm*v_norm;
else
    [Unew, Dnew, Vnew] = svd_rank_one_update1(Uold, diag(dold), Vold, u_add, v_add);
end

[Unew, Dnew, Vnew] = thinSVD(Unew, Dnew, Vnew, 1e-6);
Xnew = struct();
Xnew.U = Unew;
Xnew.V = Vnew;
Xnew.d = diag(Dnew);
Xnew.vec = Xold.vec;
