function [u, v, zBz] = Lomat(A, U, V, d, mu, eigifp_init)
% This gives the close form solution of problem
%    min <A,X>   s.t. ||X||_* - <Xi, X> <= delta

if norm(d) < 1e-8
    Xi_d = 0;
else
    Xi_d = d*(mu/norm(d));
end

[m, n] = size(A);
Ahandle = @(z) [ A*z(m+1:m+n);  A'*z(1:m) ];
Bhandle = @(z) z - [ U*(Xi_d.*(V'*z(m+1:m+n)));  V*(Xi_d.*(U'*z(1:m))) ];

% opts.DISP = 1;
opts.normA = norm(Ahandle(ones(m+n, 1)), 1);
opts.normB = norm(Bhandle(ones(m+n, 1)), 1);
opts.SIZE = m+n;
opts.initialvec = eigifp_init;
opts.tol = 1e-6;    % works only when dimension is large
[~, z] = eigifp(Ahandle, Bhandle, 1, opts);

zBz = z'*Bhandle(z);
u = z(1:m);
v = z(m+1:m+n);

