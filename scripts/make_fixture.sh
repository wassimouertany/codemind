#!/usr/bin/env bash
# Generates tests/fixtures/sample_repo — a tiny repo with a deliberate bug
# (PaymentTimeoutException escaping OrderService) plus noise that the loader
# must filter out. Every later stage is tested against this.
set -euo pipefail

FIX="tests/fixtures/sample_repo"
JAVA="$FIX/src/main/java/com/demo"

mkdir -p "$JAVA/controller" "$JAVA/service" "$JAVA/repository" "$JAVA/model"
mkdir -p "$FIX/app" "$FIX/node_modules/junk" "$FIX/target"

cat > "$JAVA/controller/OrderController.java" <<'EOF'
package com.demo.controller;

import com.demo.service.OrderService;
import com.demo.model.Order;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/orders")
public class OrderController {

    private final OrderService orderService;

    public OrderController(OrderService orderService) {
        this.orderService = orderService;
    }

    @PostMapping
    public Order createOrder(@RequestBody Order order) {
        return orderService.placeOrder(order);
    }

    @GetMapping("/{id}")
    public Order getOrder(@PathVariable Long id) {
        return orderService.findById(id);
    }
}
EOF

cat > "$JAVA/service/OrderService.java" <<'EOF'
package com.demo.service;

import com.demo.repository.OrderRepository;
import com.demo.model.Order;
import org.springframework.stereotype.Service;

@Service
public class OrderService {

    private final OrderRepository orderRepository;
    private final PaymentService paymentService;

    public OrderService(OrderRepository orderRepository, PaymentService paymentService) {
        this.orderRepository = orderRepository;
        this.paymentService = paymentService;
    }

    public Order placeOrder(Order order) {
        // BUG: PaymentTimeoutException is unchecked and never handled here,
        // so it propagates to the controller and surfaces as HTTP 500.
        paymentService.processPayment(order.getTotal());
        return orderRepository.save(order);
    }

    public Order findById(Long id) {
        return orderRepository.findById(id).orElseThrow();
    }
}
EOF

cat > "$JAVA/service/PaymentService.java" <<'EOF'
package com.demo.service;

import org.springframework.stereotype.Service;

@Service
public class PaymentService {

    private static final int TIMEOUT_MS = 3000;

    public void processPayment(double amount) {
        long elapsed = callGateway(amount);
        if (elapsed > TIMEOUT_MS) {
            throw new PaymentTimeoutException("gateway exceeded " + TIMEOUT_MS + "ms");
        }
    }

    private long callGateway(double amount) {
        return System.currentTimeMillis() % 5000;
    }
}
EOF

cat > "$JAVA/repository/OrderRepository.java" <<'EOF'
package com.demo.repository;

import com.demo.model.Order;
import org.springframework.data.jpa.repository.JpaRepository;

public interface OrderRepository extends JpaRepository<Order, Long> {
    Order findByCustomerId(Long customerId);
}
EOF

cat > "$JAVA/model/Order.java" <<'EOF'
package com.demo.model;

public class Order {
    private Long id;
    private double total;

    public Long getId() { return id; }
    public double getTotal() { return total; }
    public void setTotal(double total) { this.total = total; }
}
EOF

cat > "$FIX/app/user_service.py" <<'EOF'
"""Mirror of the Java bug in Python, for cross-language chunking tests."""

from app.user_repository import UserRepository


class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user_by_id(self, user_id: int) -> dict:
        user = self._repository.find_by_id(user_id)
        if user is None:
            raise KeyError(user_id)
        return user

    def deactivate_user(self, user_id: int) -> None:
        user = self.get_user_by_id(user_id)
        user["active"] = False
        self._repository.save(user)
EOF

cat > "$FIX/app/user_repository.py" <<'EOF'
class UserRepository:
    def __init__(self, db):
        self._db = db

    def find_by_id(self, user_id: int):
        return self._db.get(user_id)

    def save(self, user: dict) -> None:
        self._db[user["id"]] = user
EOF

# --- noise the loader must reject -------------------------------------------
echo 'module.exports={};'                 > "$FIX/node_modules/junk/index.js"
{ printf 'var a=1;'; head -c 3000 /dev/zero | tr '\0' 'x'; echo; } \
                                          > "$FIX/app/vendor.min.js"
echo 'class Compiled {}'                  > "$FIX/target/Compiled.java"
printf 'target/\n*.log\n'                 > "$FIX/.gitignore"
echo '# sample repo'                      > "$FIX/README.md"
printf "\x00\x01\x02PK\x03\x04\xff\xfe binary not code \x00" > "$FIX/app/blob.py"

echo "fixture created:"
find "$FIX" -type f | sort
