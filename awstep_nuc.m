function [u_aw, v_aw, alpha_aw] = awstep_nuc(Grad, sigma, U,  V, d, mu, m, n, zeta)
% implementation of away-step oracle 
% max <A,u_iv_i'>   s.t. X_k = sum r_iu_iv_i'; sum r_i = 1;

r = length(d);
normd = norm(d);
c = zeros(r, 1);

if normd < 1e-8
    u_aw = zeros(m, 1);
    v_aw = zeros(n, 1);
    alpha_aw = 0;
    return;
end


val_idmax = -Inf;
for i = 1:r
    gamma_i = sigma / (1-mu*d(i)/normd);
    c(i) = d(i) / gamma_i;
    val_i = gamma_i * ( U(:, i)'*Grad*V(:, i) );
    if val_i > val_idmax
        val_idmax = val_i;
        gamma_idmax = gamma_i;
        id_max = i;
        c_idmax = c(id_max);
    end
end
u_aw = U(:, id_max);
v_aw = gamma_idmax * V(:, id_max);
c_aw = c_idmax;

if sum(c) < 1
    gamma_prime = - sigma / (1 + mu*d(id_max)/normd);
    c_prime = (1 - sum(c)) * (1 + mu*d(id_max)/normd) / 2;
    val_prime = gamma_prime * val_idmax / gamma_idmax;
    c_idmax = c_idmax + (1 - sum(c)) * (1 - mu*d(id_max)/normd) / 2;
    if val_idmax <= val_prime
        v_aw = gamma_prime * V(:, id_max);
        c_aw = c_prime;
    else
        c_aw = c_idmax;
    end
end

alpha_full = c_aw / (1 - c_aw);
alpha_aw = min(alpha_full, zeta);





